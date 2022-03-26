import os
from dataclasses import dataclass
import xmltodict
from types import SimpleNamespace

class DictClass(SimpleNamespace):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            return None


@dataclass
class BoundedBox:
    xmin:int
    ymin:int
    xmax:int
    ymax:int

    def __init__(self, xmin, ymin, xmax, ymax):
        ensure_type = lambda it: it if isinstance(it, int) else int(it)
        self.xmin = ensure_type(xmin)
        self.ymin = ensure_type(ymin)
        self.xmax = ensure_type(xmax)
        self.ymax = ensure_type(ymax)

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def center(self):
        return (self.xmax + self.xmin)/2, (self.ymax+self.ymin)/2

@dataclass
class NamedBoundedBox:
    name:str
    box: BoundedBox

@dataclass
class VocFile:
    def __init__(self, image_file, anno_file):
        self.image_file = image_file
        self.anno_file = anno_file
        with open(self.anno_file ,"rb") as f:
            annotation = xmltodict.parse(f)
        
        def recursive_parse(node):
            if isinstance(node, dict):
                node = {k: recursive_parse(v) for k,v in node.items()}
                return DictClass(**node)
            if isinstance(node, list):
                return [recursive_parse(it) for it in node]
            return node

        root = recursive_parse(annotation)
        self.annotation = root.annotation

    @property
    def type(self):
        return self.annotation.object.name

    @property
    def bounded_box(self):
        box = self.annotation.object.bndbox
        return NamedBoundedBox(self.type, self.__named_to_bounded_box(box))

    @property
    def parts(self):
        if not self.annotation.object.part:
            return []
        return [NamedBoundedBox(it.name,self.__named_to_bounded_box(it.bndbox)) 
                    for it in self.annotation.object.part]
    
    def __named_to_bounded_box(self, named_tuple):
        return BoundedBox(named_tuple.xmin, named_tuple.ymin, named_tuple.xmax, named_tuple.ymax)

    @property
    def size(self):
        return self.annotation.size
        
                
class VocDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir
    
    def get_file(self, filename):
        return VocFile(
            os.path.join(self.base_dir,"JPEGImages",filename+".jpg"),
            os.path.join(self.base_dir,"Annotations",filename+".xml")
        )
