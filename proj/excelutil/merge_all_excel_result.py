import xlrd
import xlwt
import glob
import os
import shutil
def process_one_excel(j,path,wr_sheet):
    # readbook = xlrd.open_workbook(path)
    # rd_sheet = readbook.sheet_by_index(0)
    # nrows = rd_sheet.nrows
    # for i in range(1,nrows):
    #     dice=rd_sheet.cell_value(i,0)
    #     wr_sheet.write(i,j,(dice))
    i=0
    for line in open(path):
        # print line,  #python2 用法
        print(line)
        term=line.split("\t")
        value=float(term[0])
        wr_sheet.write(i, j, value)
        i=i+1

def merge_all_result(dir):
    writebook = xlwt.Workbook(encoding = 'utf-8')
    wr_sheet = writebook.add_sheet("all")
    all_file=glob.glob(dir+"/EvaluateResultsResample/*/"+"*dice.xls",recursive=True)
    for i,path in enumerate(all_file):
        process_one_excel(i,path,wr_sheet)

    if os.path.exists(dir+r"/merge_all.xls"):
        os.remove(dir+r"/merge_all.xls")
    writebook.save(dir+r"/merge_all.xls")


if __name__=="__main__":
    # merge_all_result("../../data_20_60/MRI_test/205/warp_atalas_mr_mr/lab")
    merge_all_result("../../outputs/ct-mr-205/test/atlas_wise/")
    # merge_all_result("../../outputs/mr-ct-205/test/atlas_wise/")
    # merge_all_result("../../data_20_60/MRI_test/zip-mr-test-label_crop_reg_out_ct_mr")
    # merge_all_result("../../data_20_60/CT_test/zip-ct-test-label_crop_reg_out_ct_ct")
    # merge_all_result("../../data_20_60/CT_test/zip-ct-test-label_crop_reg_out_mr_ct")
