models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    model = models[name](config)
    return model


from . import nerf, bnerf, ssnerf1, ssnerf2, neus, geometry, texture, shutter_speed
