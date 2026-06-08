"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

Copyright (c) 2023 WonJun Moon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
import vocab
from basic_utils import load_jsonl, l2_normalize_np_array
from tensor_utils import pad_sequences_1d
from span_utils import span_xx_to_cxw
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

logger = logging.getLogger(__name__)


class CGDETR_StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, domain, data_path, v_feat_dirs, a_feat_dirs,
                 q_feat_dir, q_feat_type="last_hidden_state", v_feat_types="clip",
                 a_feat_types="pann", max_q_l=32, max_v_l=75, max_a_l=75,
                 ctx_mode="video", normalize_v=True, normalize_t=True, clip_len=2, max_windows=5, 
                 span_loss_type="l1", dset_domain=None, load_labels=True, 
                 is_train=False, shift_prob=0.8, margin_frames=3):
        self.dset_name = dset_name
        self.data_path = data_path
        self.domain = domain
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.a_feat_dirs = a_feat_dirs \
            if isinstance(a_feat_dirs, list) else [a_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.v_feat_types = v_feat_types
        self.a_feat_types = a_feat_types
        
        if max_v_l == -1:
            max_v_l = 100000000
        if max_a_l == -1:
            max_a_l = 100000000
        if max_q_l == -1:
            max_q_l = 100
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.max_a_l = max_a_l
        
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.use_audio = "audio" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.load_labels = load_labels

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

        if self.dset_name == 'tvsum' or self.dset_name == 'youtube_highlight':
            new_data = []
            for d in self.data:
                if d['domain'] == self.domain:
                    new_data.append(d)
            self.data = new_data

        self.use_glove = 'glove' in q_feat_dir
        if self.use_glove:
            self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
            self.vocab.itos.extend(['<unk>'])
            self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
            self.vocab.vectors = torch.cat(
                (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
            self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)
        
        # === 最適化: TEFの事前計算キャッシュ ===
        self.tef_cache = {}
        if self.use_tef:
            max_len = max(self.max_v_l, self.max_a_l)
            # 1 〜 最大長 までのTEFを全て事前に計算してメモリに保持
            for l in range(1, max_len + 1):
                tef_st = torch.arange(0, l, 1.0) / l
                tef_ed = tef_st + 1.0 / l
                self.tef_cache[l] = torch.stack([tef_st, tef_ed], dim=1)
        # === 追加：時間シフト用の設定 ===        
        self.is_train = is_train
        self.shift_prob = shift_prob
        self.margin_frames = margin_frames
        

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        original_meta = self.data[index]
        # 1. 辞書の浅いコピーを作成（高速）
        meta = {k: v for k, v in original_meta.items()}
        
        # 2. シフト処理で値が書き換わるリストだけを新しく作り直す（高速）
        if "relevant_windows" in meta:
            meta["relevant_windows"] = [[w[0], w[1]] for w in original_meta["relevant_windows"]]
        if "relevant_clip_ids" in meta:
            meta["relevant_clip_ids"] = list(original_meta["relevant_clip_ids"])

        model_inputs = dict()

        if self.use_glove:
            model_inputs["query_feat"] = self.get_query(meta["query"])
        else:
            model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)
            
        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_audio:
            #assert self.a_feat_types is not None, f"use_audio is {self.use_audio}, but a_feat_types is {self.a_feat_types}."
            model_inputs["audio_feat"] = self._get_audio_feat_by_vid(meta["vid"])
            ctx_l_a = len(model_inputs["audio_feat"])
            # Sometimes, audio features is longer than video features because the length of video is not necessarily 2:30.
            if ctx_l < ctx_l_a:
                model_inputs["audio_feat"] = model_inputs["audio_feat"][:ctx_l]
                ctx_l_a = ctx_l
            elif ctx_l > ctx_l_a:
                if self.use_video:
                    model_inputs["video_feat"] = model_inputs["video_feat"][:ctx_l_a] # TODO: Sometimes, audio length is not equal to video length.
                ctx_l = ctx_l_a
        else:
            ctx_l_a = self.max_a_l

        # ========================================================
        # 時間軸のランダムシフト (Data Augmentation)
        # ========================================================
        if self.is_train and random.random() < self.shift_prob:
            # GTの最小開始時刻と最大終了時刻を取得
            st_min = min([w[0] for w in meta["relevant_windows"]])
            ed_max = max([w[1] for w in meta["relevant_windows"]])
            
            st_frame_min = int(math.floor(st_min / self.clip_len))
            ed_frame_max = int(math.ceil(ed_max / self.clip_len))
            
            # 映像を使わない audio_tef モードに対応するため、有効な最大長を切り替える
            effective_max_l = self.max_v_l if self.use_video else self.max_a_l

            # 左シフト(削り)と右シフト(パディング)の限界を計算
            max_left = max(0, st_frame_min - self.margin_frames)
            max_right = max(0, effective_max_l - ed_frame_max - self.margin_frames)
            
            if max_left > 0 or max_right > 0:
                std_dev = max(max_left, max_right) / 2.0 
                shift = int(random.gauss(mu=0, sigma=std_dev))
                shift = max(-max_left, min(max_right, shift))

                if shift != 0:
                    # 映像特徴量のシフト (使用する場合のみ)
                    if self.use_video and "video_feat" in model_inputs:
                        v_feat = model_inputs["video_feat"]
                        if shift > 0:
                            v_feat = F.pad(v_feat, (0, 0, shift, 0))[:self.max_v_l]
                        else:
                            v_feat = v_feat[-shift:]
                        model_inputs["video_feat"] = v_feat
                    
                    # 音声特徴量のシフト
                    if self.use_audio and "audio_feat" in model_inputs:
                        a_feat = model_inputs["audio_feat"]
                        if shift > 0:
                            a_feat = F.pad(a_feat, (0, 0, shift, 0))[:self.max_a_l]
                        else:
                            # 【安全弁】シフトで削りすぎて配列が空（長さ0）になるのを防ぐ
                            if -shift >= len(a_feat):
                                a_feat = a_feat[-1:] # 最低でも最後の1フレームを残す
                            else:
                                a_feat = a_feat[-shift:]
                        model_inputs["audio_feat"] = a_feat

                    # 時刻の同期更新
                    shift_sec = shift * self.clip_len
                    for i in range(len(meta["relevant_windows"])):
                        meta["relevant_windows"][i][0] += shift_sec
                        meta["relevant_windows"][i][1] += shift_sec
        # ========================================================

        # 最終的なデータ長（ctx_l）を現在の特徴量から厳密に再計算
        if self.use_video and "video_feat" in model_inputs:
            ctx_l = len(model_inputs["video_feat"])
        elif self.use_audio and "audio_feat" in model_inputs:
            ctx_l = len(model_inputs["audio_feat"])
        else:
            ctx_l = self.max_v_l

        # 【絶対防御】何があっても ctx_l を 0 にしない、かつキャッシュの範囲内に収める
        max_cache_len = len(self.tef_cache) - 1
        ctx_l = max(1, min(ctx_l, max_cache_len))

        if self.use_tef:
            tef = self.tef_cache[ctx_l] # これで絶対に KeyError: 0 は起きません
            if self.use_video:
                model_inputs["video_feat"] = torch.cat([model_inputs["video_feat"], tef], dim=1)
            else:
                # audio_tef モードの場合、モデルの仕様に合わせて video_feat の位置に TEF を格納
                model_inputs["video_feat"] = tef

        # 4. ラベル（Span, Saliency）の取得
        if self.load_labels:
            if "relevant_windows" in meta:
                # 区間予測用のラベル
                model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)
                # ハイライト（Saliency）用のラベル（Clotho-Moment, CASTELLA 共通で複数区間対応ロジックを使用）
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                    self.get_saliency_labels_multi_windows(meta["relevant_windows"], meta["duration"], ctx_l, max_n=4)
        
        model_inputs["vid"] = meta["vid"]
        model_inputs["qid"] = meta["qid"]
        return dict(meta=meta, model_inputs=model_inputs)

    def get_query(self, query):
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        return self.embedding(word_inds)


    def get_saliency_labels_multi_windows(self, windows, duration, ctx_l, max_n=2):
        """
        複数の正解区間(windows)をすべてポジティブとして扱い、max_n個のペアを生成する
        """
        clip_len = duration / ctx_l
        score_array = np.zeros(ctx_l)
        pos_pool = set()

        # 1. すべての正解区間についてスコアを1にし、ポジティブなフレームの集合を作る
        for window in windows:
            gt_st = int(window[0] / clip_len)
            gt_ed = max(0, min(int(window[1] / clip_len), ctx_l) - 1)
            if gt_st > gt_ed:
                gt_st = gt_ed
            
            score_array[gt_st:gt_ed + 1] = 1
            for i in range(gt_st, gt_ed + 1):
                pos_pool.add(i)
                
        pos_pool = list(pos_pool)

        # 2. ポジティブサンプルの抽出 (全正解区間の中からランダムに max_n 個選ぶ)
        if len(pos_pool) == 0:
            pos_clip_indices = [0] * max_n
        elif len(pos_pool) >= max_n:
            pos_clip_indices = random.sample(pos_pool, k=max_n) # 重複なし
        else:
            pos_clip_indices = random.choices(pos_pool, k=max_n) # フレームが少ない場合は重複を許容

        # 3. ネガティブサンプルの抽出 (正解区間以外の完全に無関係なフレームから選ぶ)
        neg_pool = list(set(range(ctx_l)) - set(pos_pool))
        if len(neg_pool) >= max_n:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        elif len(neg_pool) > 0:
            neg_clip_indices = random.choices(neg_pool, k=max_n)
        else:
            # 例外: 動画全体が正解区間になってしまっている場合
            neg_clip_indices = pos_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array


    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows


    def _get_query_feat_by_qid(self, qid):
        if self.dset_name == 'tvsum' or self.dset_name == 'youtube_highlight':
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            q_feat = np.load(q_feat_path)
            return torch.from_numpy(q_feat['token']) if self.dset_name == 'tvsum' else torch.from_numpy(q_feat['last_hidden_state'])
        
        elif self.dset_name == 'tacos':
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)

        # --追加 M2D-CLAP テキスト特徴量用の処理--
        elif self.dset_name == 'castella' and self.q_feat_type == 'last_hidden_state':
            q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
            # -- 追加：ファイルが存在しない場合はダミー特徴量を生成
            if not exists(q_feat_path):
                logger.warning(f"Castella text file missing, using dummy features for: {qid}")
                q_feat = np.zeros((1, 768), dtype=np.float32)
            else:
                q_feat = np.load(q_feat_path)["last_hidden_state"].astype(np.float32)
            if q_feat.ndim == 1:
                q_feat = np.expand_dims(q_feat, axis=0) # (D, ) -> (1, D)
            elif q_feat.ndim == 2 and q_feat.shape[0] != 1:
                if q_feat.shape[1] == 768:
                    pass
                else:
                    q_feat = q_feat.reshape(1, -1)
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)

        else:
            # QVhighlight dataset
            if "subs_train" in self.data_path: # for pretrain
                vid = "_".join(qid.split("_")[:-1])
                subid = qid.split("_")[-1]
                q_feat_path = join(self.q_feat_dir, f"{vid}/{subid}.npz")
            else:
                q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")

                # --追加：ファイルが存在しない場合はダミー特徴量を生成--
                if not exists(q_feat_path):
                    logger.warning(f"Text file missing, using dummy features for: {qid}")
                    q_feat = np.zeros((1, 768), dtype=np.float32)
                else:
                    q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
        
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            _feat_path = join(_feat_dir, f"{vid}.npz")
            _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
            if not exists(_feat_path):
                _feat_path = join(_feat_dir, f"{vid}.npy")
                _feat = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
            else:
                _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
            if self.normalize_v:
                _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)

    def _get_audio_feat_by_vid(self, vid):
        a_feat_list = []
        for _feat_dir in self.a_feat_dirs:
            if self.dset_name == 'qvhighlight' or self.dset_name == 'qvhighlight_pretrain':
                if self.a_feat_types == "pann":
                    _feat_path = join(_feat_dir, f"{vid}.npy")
                    _feat = np.load(_feat_path)[:self.max_a_l].astype(np.float32)
                else:
                    raise NotImplementedError
                _feat = l2_normalize_np_array(_feat) # normalize?
                a_feat_list.append(_feat)
            elif self.dset_name in ['clotho-moment', 'unav100-subset', 'tut2017', 'castella']:
                # --M2D-CLAPもCLAPと同じ動きをするように条件を追加--
                if self.a_feat_types in ["clap", "m2dclap"]:
                    _feat_path = join(_feat_dir, f"{vid}.npz")
                    # --ファイルが存在しない場合はダミーを生成して返す
                    if not exists(_feat_path):
                        logger.warning(f"Audio file missing, using dummy features for: {vid}")
                        _feat = np.zeros((300, 768), dtype=np.float32)
                    else:
                        _feat = np.load(_feat_path)["features"][:self.max_a_l].astype(np.float32)
                else:
                    raise NotImplementedError
                _feat = l2_normalize_np_array(_feat) # normalize?
                a_feat_list.append(_feat)
            else:
                raise NotImplementedError
        
        # some features are slightly longer than the others
        min_len = min([len(e) for e in a_feat_list])
        a_feat_list = [e[:min_len] for e in a_feat_list]
        a_feat = np.concatenate(a_feat_list, axis=1)
        return torch.from_numpy(a_feat)  # (Lv, D)


def cg_detr_start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        if k == 'qid':
            batched_data[k] = [e["model_inputs"][k] for e in batch]
            continue
        if k == 'vid':
            batched_data[k] = [e["model_inputs"][k] for e in batch]
            continue
        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def cg_detr_prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        vid=batched_model_inputs["vid"],
        qid=batched_model_inputs["qid"],
    )

    if "audio_feat" in batched_model_inputs:
        model_inputs["src_aud"] = batched_model_inputs["audio_feat"][0].to(device, non_blocking=non_blocking)
        model_inputs["src_aud_mask"] = batched_model_inputs["audio_feat"][1].to(device, non_blocking=non_blocking)

    targets = {}

    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
        targets["relevant_clips"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
