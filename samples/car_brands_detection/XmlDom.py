from xml.dom.minidom import parse
import xml.dom.minidom

class XmlParser(object):
    def __init__(self):
        self.path = ""
        self.boxes = []

    def parseBox(self,box_ele):
        name=""
        xmin=""
        ymin=""
        xmax =""
        ymax = ""
        menbers = box_ele.childNodes
        for menber in menbers:
            if (menber.firstChild != None and menber.nodeName == "name"):
                name_ele = menber.getElementsByTagName("name")
                name = menber.childNodes[0].data
            if (menber.firstChild != None and menber.nodeName == "bndbox"):
                bndboxs_ele = menber.getElementsByTagName("bndbox")
                bndboxs = menber.childNodes
                for bndbox in bndboxs:
                    if (bndbox.firstChild != None and bndbox.nodeName == "xmin"):
                        xmin_ele = bndbox.getElementsByTagName("xmin")
                        xmin = bndbox.childNodes[0].data
                    if (bndbox.firstChild != None and bndbox.nodeName == "ymin"):
                        ymin_ele = bndbox.getElementsByTagName("ymin")
                        ymin = bndbox.childNodes[0].data
                    if (bndbox.firstChild != None and bndbox.nodeName == "xmax"):
                        xmax_ele = bndbox.getElementsByTagName("xmax")
                        xmax = bndbox.childNodes[0].data
                    if (bndbox.firstChild != None and bndbox.nodeName == "ymax"):
                        ymax_ele = bndbox.getElementsByTagName("ymax")
                        ymax = bndbox.childNodes[0].data
        return (name,xmin,ymin,xmax,ymax)


    def parse(self,file_path):
        DOMTree = xml.dom.minidom.parse(file_path)
        collection = DOMTree.documentElement
        path_ele = collection.getElementsByTagName("path")
        self.path = path_ele[0].childNodes[0].data
        boxes_eles = collection.getElementsByTagName("object")
        for box_ele in boxes_eles:
            self.boxes.append(self.parseBox(box_ele))


# parser = XmlParser()
# parser.parse("D:\\Repo\\AI\\Datasets\\test_car_masks\\benz11111.xml")
# path = parser.path
# boxes = parser.boxes
#
# print("path",path)
# for box in boxes:
#     print(box[0],box[1],box[2],box[3],box[4])



