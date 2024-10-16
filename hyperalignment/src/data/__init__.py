from .image_caption_datasets import *
from .embedding_datasets import *
from .classification_datasets import *


def init_indices_loader(args, dataset):
    num_samples = len(dataset)
    batch_size = args.batch_size
    total_dataset_indices = [x for x in range(num_samples)]

    while True:
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = total_dataset_indices[start_idx : end_idx]
            yield batch_indices


def init_encoder_loader(args, dataset):
    num_encoders = sum([len(v) for v in dataset.image_encoder_data.values()])
    batch_size = args.encoder_batch_size

    assert batch_size <= min([len(v) for v in dataset.image_encoder_data.values()]), "Incorrect sampling size for encoders!"
    total_encoder_indices = [x for x in range(num_encoders)]

    # get info about the embedding dim of the sampled encoders too
    total_dim_store = [[k for _ in range(len(v))] for k, v in dataset.image_encoder_data.items()]
    total_dim_store = [y for x in total_dim_store for y in x]
    # check if all the dims are the same (mandatory)

    while True:
        for start_idx in range(0, num_encoders, batch_size):
            end_idx = min(start_idx + batch_size, num_encoders)
            encoder_indices = total_encoder_indices[start_idx : end_idx]

            encoder_dims = total_dim_store[start_idx : end_idx]
            assert set(encoder_dims) == {encoder_dims[0]}, "All encoders in the batch are not of the same dimension => future error while concating!"

            yield encoder_indices, encoder_dims
