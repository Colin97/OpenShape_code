from . import Minkowski, ppat

def make(config):
    if config.model.name == "MinkowskiFCNN":
        model = Minkowski.MinkowskiFCNN(config)
    elif config.model.name == "MinkResNet":
        model = Minkowski.MinkResNet(config)
    elif config.model.name == "MinkResNet34":
        model = Minkowski.MinkResNet34(config)
    elif config.model.name == "MinkResNet11":
        model = Minkowski.MinkResNet11(config)
    elif config.model.name == "MinkowskiFCNN_small":
        model = Minkowski.MinkowskiFCNN_small(config)
    elif config.model.name == "PointBERT":
        model = ppat.make(config)
    elif config.model.name == "DGCNN":
        from . import dgcnn
        model = dgcnn.make(config)
    elif config.model.name == "PointNeXt":
        from . import pointnext
        model = pointnext.make(config)
    elif config.model.name == "PointMLP":
        from . import pointmlp
        model = pointmlp.make(config)
    elif config.model.name == "PointNet":
        from . import pointnet
        model = pointnet.make(config)
    else:
        raise NotImplementedError("Model %s not supported." % config.model.name)
    return model
