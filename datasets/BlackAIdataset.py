import json
from pathlib import Path

import torch
import torch.utils.data as data

from .loader import VideoLoader
import os
from collections import defaultdict
import tqdm
import random
import copy
from tqdm import tqdm
from .Class_define import ucf_101_class,kinetics_700_labels

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations

def _reverse_sample(sample):
	sample_reverse = copy.deepcopy(sample)
	sample_reverse['frame_indices'].reverse()
	return sample_reverse


def _mirror_sample(sample):
	sample_mirror = copy.deepcopy(sample)
	sample_mirror['mirror'] = True
	return sample_mirror

def make_data(root_path, annotation_path, allow_reverse, mirror, balance, balance_proportions,subset, sample_duration, required_overlap, shuffle):
    assert (subset in ['train', 'val', 'test'], 'subset "{}" is not "train", "val" or "test"'.format(subset))

    changevalue = {'validation': 'val', 'train' : 'train'}
    subset = changevalue[subset]
    frame_format = lambda x: str(x).zfill(4) + '.png'

    data = []
    action_list = _load_list( annotation_path)
    local_track_action_dict = defaultdict(list)

    for action in action_list:

        local_track_action_dict[action[0]].append(action[1:])

    for local_track_path, actions in tqdm(local_track_action_dict.items(), desc='Creating a dataset'):

        # sort the list of action by the starting frame
        actions.sort(key=lambda a: a[1])
        if subset != actions[0][-2]: continue
        frame_dir = os.path.join(root_path, local_track_path)
        files = os.listdir(frame_dir)

        #  split the frames into samples
        for i in range(len(files) // sample_duration):
            use_sample = True
            sample_start = i * sample_duration + 1
            sample_end = min((i + 1) * sample_duration, len(files))

            # make sure all the frames in the sample are saved properly, if they're not ignore the sample
            frame_indices = list(range(sample_start, sample_end + 1))
            for index in frame_indices:
                frame_path = os.path.join(root_path, local_track_path, frame_format(index))
                if not os.path.exists(frame_path):
                    use_sample = False
                    break

            if not use_sample: continue

            # count the number of frames in sample that are part of the action
            action_label = 0
            for (label, start, end, frame, subset, hard_data) in actions:
                action_overlap = max(0, end - sample_start + 1 if sample_start > start else sample_end - start + 1)
                if action_overlap > sample_duration * required_overlap:
                    action_label = label
                    break

            sample = {'path': frame_dir,
                      'label': action_label,
                      'frame_indices': frame_indices,
                      'mirror': False}

            data.append(sample)

    reverse_data = []
    if allow_reverse:
        for sample in tqdm(data, desc='Reversing the data'):
            reverse = True if sample['label'] != 0 else random.random() > 0.5
            if reverse:
                reverse_data.append(_reverse_sample(sample))
    data.extend(reverse_data)

    mirror_data = []
    if mirror:
        for sample in tqdm(data, desc='Adding mirror flip'):
            mirror_data.append(_mirror_sample(sample))
    data.extend(mirror_data)

    labels = set([i['label'] for i in data])

    count_labels = {label: sum([1 for i in data if i['label'] == label]) for label in labels}
    print('Dataset is generated. Label counts are:', count_labels)

    if balance:
        required = min(count_labels.values())

        if balance_proportions is not None:
            sampling_probabilities = {label: (required / value) * balance_proportions[label] for label, value in
                                      count_labels.items()}
        else:
            sampling_probabilities = {label: required / value for label, value in count_labels.items()}

        balanced_data = []
        for sample in tqdm(data, desc='Balancing dataset'):
            if random.random() < sampling_probabilities[sample['label']]:
                balanced_data.append(sample)

        data = balanced_data
        count_labels = {label: sum([1 for i in data if i['label'] == label]) for label in labels}
        print('Balanced dataset is generated. Label counts are:', count_labels)
    if shuffle:
        random.shuffle(data)
    return data


def _load_list(path):
	print('Loading a saved annotation file from {}'.format(path))
	with open(path, 'r') as f:
		mylist = []
		for line in f:
			line = line.split('\n')[0]
			splitline = line.split(' ')
			#some dataset been label "is hard dataset?"
			if len(splitline) == 7:
				mylist.append((splitline[0], int(splitline[1]), int(splitline[2]), int(splitline[3]), int(splitline[4]), splitline[5], int(splitline[6])))
			else:
				mylist.append((splitline[0], int(splitline[1]), int(splitline[2]), int(splitline[3]), int(splitline[4]), splitline[5], 0))
	return mylist

class BlackAIdataset_(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        # with annotation_path.open('r') as f:
        #     data = json.load(f)

        allow_reverse = False
        mirror = False
        shuffle = False
        if subset == 'train':
            allow_reverse = True
            mirror = True
            shuffle = True


        BlackAI_dataformat = make_data(root_path, annotation_path, allow_reverse=allow_reverse, mirror = mirror, balance = True, balance_proportions = {0:2, 1:1}, subset = subset, sample_duration = 16, required_overlap = 0.7, shuffle = shuffle)
        video_ids = []
        video_paths = []
        annotations = []
        for it in BlackAI_dataformat:
            video_ids.append(os.path.basename(it['path']))
            video_paths.append(it['path'])
            if it['label'] == 0:
                label = 'walk'
            else:
                label = 'action'
            annotations.append({'label':label, 'segment':[it['frame_indices'][0], it['frame_indices'][len(it['frame_indices']) - 1]]})


        #video_ids, video_paths, annotations = _load_list( annotation_path)
        # video_ids, video_paths, annotations = get_database(
        #     data, subset, root_path, video_path_formatter)
        class_to_idx = {'walk':0, 'action':1}
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1



            video_path = video_paths[i]
            if not os.path.exists(video_path):
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue
            if  segment[1] > segment[0]:
                frame_indices = list(range(segment[0], segment[1]))
            else:
                frame_indices = list(range(segment[1], segment[0]))
                list.reverse(frame_indices)


            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)