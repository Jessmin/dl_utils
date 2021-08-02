import json
import xlrd
import xlwt

def achieve_data(file_path):
    try:
        data = xlrd.open_workbook(file_path)
        return data
    except Exception as e:
        print("excel表格读取失败：%s" % file_path)
        return None


def excel2json():
    file_path = './test.xlsx'
    data = achieve_data(file_path)
    if data is not None:
        # 抓取所有sheet页的名称
        worksheets = data.sheet_names()
        print("包含的表单:")
        for index, sheet in enumerate(worksheets):
            print(index, sheet)
        choose = 0
        table = data.sheet_by_index(int(choose))
        # 获取到数据的表头
        titles = table.row_values(0)
        result = {}
        result[0] = []
        # excel文件表有 10196 行，所以做10196次循环
        for i in range(1, table.nrows):
            row = table.row_values(i)
            tmp = {}
            for index, title in enumerate(titles):
                tmp[index] = row[index]
            result[0].append(tmp)
        with open("middle_school.json", 'w') as f:
            # json.dump(result, f)
            f.write(json.dumps(result, ensure_ascii=False))


excel2json()