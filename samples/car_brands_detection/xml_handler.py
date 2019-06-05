import xml.sax


class XmlHandler(xml.sax.ContentHandler):

    def __init__(self):

        # self.folder =""
        # self.filename = ""
        self.currentData = ""
        self.annotation = ""
        self.path = ""
        self.object = ""
        self.name = ""
        self.bndbox = ""
        self.xmin = ""
        self.ymin = ""
        self.xmax = ""
        self.ymax = ""
        self.map = {"annotation":1,"path":1,"object":1,"bndbox":1,"name":1,}
        self.characters_map = {"path":1,"name":1,"xmin":1,"ymin":1,"xmax":1,"ymax":1}


    def startElement(self, tag, attributes):
        a = tag
        b = attributes
        if tag in self.map:
            self.CurrentData = tag
        # if tag == "annotation":
        #     self.currentData = tag
        # elif tag == "path":
        #     self.currentData = tag
        # elif tag == "object":
        #     self.currentData = tag
        # elif tag == "bndbox":
        #     self.currentData = tag
        # elif tag == "name":
        #     self.currentData = tag

        # 元素结束调用

    def endElement(self, tag):
        c=tag
        if self.currentData == "\t" or "\n":
            pass
        if self.currentData == "name":
            print("name:", self.name)
        elif self.currentData == "xmin":
            print("xmin:", self.xmin)
        elif self.currentData == "ymin":
            print("ymin:", self.ymin)
        elif self.currentData == "xmax":
            print("xmax:", self.xmax)
        elif self.currentData == "ymax":
            print("ymax:", self.ymax)
        self.CurrentData = ""


    def characters(self, content):
        d = content
        currentData = self.currentData
        #if self.currentData  in self.characters_map:
        if self.currentData =="path":
            self.path = content
        elif self.currentData == "name":
            self.name = content
        elif self.currentData == "xmin":
            self.xmin = content
        elif self.currentData == "ymin":
            self.ymin = content
        elif self.currentData == "xmax":
            self.xmax = content
        elif self.currentData == "ymax":
            self.ymax = content

    # import os
    # import sys
    # sys.path.append()
    # from xml_handler import XmlHandler
if (__name__ == "__main__"):
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # 关闭命名空间
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # 重写 ContextHandler
    Handler = XmlHandler()
    parser.setContentHandler(Handler)

    parser.parse("D:\Repo\AI\Datasets\\test_car_masks\\benz01.xml")



