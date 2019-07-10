import json
import copy
import functools

import torch
from torch.utils.data.dataloader import default_collate

from .videodataset import VideoDataset
from .utils import get_class_labels, get_video_ids_and_annotations


def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    return default_collate(batch_clips), batch_targets


class VideoDatasetMultiClips(VideoDataset):

    def __loading(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
            segments.append([clip_frame_indices[0], clip_frame_indices[-1] + 1])

        return clips, segments

    def __getitem__(self, index):
        path = self.data[index]['video']

        video_frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)

        clips, segments = self.__loading(path, video_frame_indices)

        video_id = self.data[index]['video_id']

        return clips, [(video_id, s) for s in segments]