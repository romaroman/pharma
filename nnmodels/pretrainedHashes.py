# from redis import Redis
#
# from nearpy import Engine
# from nearpy.storage import RedisStorage
#
# import torch
# from torch import nn
# import torchvision
#
# from nnmodels.hash import ResNet18Hash
# from nnmodels.datasets import PharmaPackDataset, get_train_validation_data_loaders
# from nnmodels import config
# import utils
#
#
# def create(train_loader, engine, model):
#     img_index = 0
#     with torch.no_grad():
#         model.eval()
#
#         for batch_idx, (data, target) in enumerate(train_loader):
#
#             v256 = model(*data)  # v1024, v512,
#             for vector, tc in zip(v256.numpy(), target.numpy()):
#                 engine.store_vector(v=vector, data=f"{str(tc)}_{utils.zfill_n(img_index, 9)}")
#                 img_index += 1
#
#
# def evaluate(test_loader, engine, model):
#     with torch.no_grad():
#         model.eval()
#
#         for batch_idx, (data, target) in enumerate(test_loader):
#
#             v256 = model(*data)
#             for vector, tc in zip(v256.numpy(), target.numpy()):
#                 neighbours = engine.neighbours(vector)
#
#
# if __name__ == '__main__':
#     model = ResNet18Hash(torchvision.models.resnet50(pretrained=True, num_classes=1000))
#     cuda = torch.cuda.is_available()
#
#     redis_db = Redis(host='localhost', port=6379, db=0)
#     redis_db.flushall()
#     engine = Engine(256, storage=RedisStorage(redis_db))
#
#     train_loader, test_loader = get_train_validation_data_loaders(PharmaPackDataset(config.source_dir))
#
#     create(train_loader, engine, model)
#
#     evaluate(test_loader, engine, model)
