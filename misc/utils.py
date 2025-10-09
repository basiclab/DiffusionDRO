import json

import click


class CommandAwareConfig(click.Command):
    def invoke(self, ctx):
        """Load config from file and overwrite by command line arguments."""
        config_file = ctx.params["config"]
        if config_file is None:
            return super(CommandAwareConfig, self).invoke(ctx)
        with open(config_file) as f:
            configs = json.load(f)
        for param in ctx.params.keys():
            if ctx.get_parameter_source(param) != click.core.ParameterSource.DEFAULT:
                continue
            if param != "config" and param in configs:
                ctx.params[param] = configs[param]
        return super(CommandAwareConfig, self).invoke(ctx)


class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and k not in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x) if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, EasyDict):
            value = EasyDict(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, *args):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, *args)
