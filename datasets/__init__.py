datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config):
    print(f"name {name} config {config}")
    dataset = datasets[name](config)
    return dataset


from . import blender, colmap, dtu
