import xlrd
import xlwt
import glob
import os
from dirutil.helper import mkdir_if_not_exist

def read_excel(path):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_index(0)
    rownum=sheet.nrows
    colnum=sheet.ncols
    dict={}
    for i in range(colnum):
        list=[]
        for j in range(1,rownum):
           list.append(sheet.cell_value(j,i))
        dict[sheet.cell_value(0,i)]=list
    return dict
def write_excel(path,map):
    writebook = xlwt.Workbook(encoding = 'utf-8')
    sheet = writebook.add_sheet("all")

    for i,key in enumerate(map.keys()):
        list=map[key]
        sheet.write(0,i,key)
        for j,item in enumerate(list):
            sheet.write(j+1,i,item)
    if os.path.exists(path):
        os.remove(path)
    writebook.save(path)

def outpu2excel(path, id, array):

    if os.path.exists(path):
        map=read_excel(path)
    else:
        mkdir_if_not_exist(os.path.dirname(path))
        map={}
    map[id]=array
    write_excel(path,map)

if __name__ == "__main__":
    outpu2excel('../../outputs/result/result.xls', 'mmwhs-ct-mr-fold-4-ds', [1.0, 0.2, 3])
    outpu2excel('../../outputs/result/result.xls', 'mmwhs-ct-mr-fold-2-hd', [1.0, 0.2, 3])
    outpu2excel('../../outputs/result/result.xls', 'mmwhs-ct-mr-fold-3', [1.0, 0.2, 3])
