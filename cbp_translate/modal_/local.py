from contextlib import contextmanager
from pathlib import Path


SHARED = Path("/tmp/cbp-translate")
SHARED.mkdir(exist_ok=True)


class Container:
    def __init__(self):
        self.__enter__()

    def __enter__(self):
        raise NotImplementedError()


class Result:
    def __init__(self, value):
        self.value = value

    def get(self):
        ret = self.value
        self.value = None
        return ret


class Function:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, cls):
        self.f = self.f.__get__(obj, cls)
        return self

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def call(self, *args, **kwargs):
        return self(*args, **kwargs)

    def map(self, *iterators, kwargs={}):
        from tqdm import tqdm

        for args in tqdm(zip(*iterators)):
            yield self.call(*args, **kwargs)

    def spawn(self, *args, **kwargs):
        return Result(self.call(*args, **kwargs))


class Stub:
    def function(self, *args, **kwargs):
        def wrapper(f):
            return Function(f)

        return wrapper

    def asgi(self, *args, **kwargs):
        def wrapper(f):
            return f

        return wrapper

    @contextmanager
    def run(self, *args, **kwargs):
        yield


stub = Stub()
hf_secret = object()
deepl_secret = object()
nemo_secret = object()
volume = object()
cpu_image = object()
gpu_image = object()
