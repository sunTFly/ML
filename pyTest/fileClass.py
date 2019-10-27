import pandas as pd
import shutil
import os, sys

data = pd.read_excel("下载地址1.xlsx")
data['姓名'] = '+' + data['姓名'].astype(str)
data['工号'] = '+' + data['工号'].astype(str)
data['标题'] = '_' + data['标题'].astype(str)
data["名称"] = data["地市"] + data["姓名"] + data["工号"] + data["标题"]
data = pd.DataFrame.drop(data, ["地市", "姓名", "工号", "标题"], axis=1)


path = 'D:/test/file/'  # 新建文件夹所在路径
path1 = 'D:/test/视频/'
path2 = 'D:/test/文档/'


def MkDir(data):
    dirs = data.drop_duplicates(['名称'])
    for dir in dirs.itertuples():
        file_name = path + str(dir[4])
        os.mkdir(file_name)
        for v in data.itertuples():
            if v[4] == dir[4]:
                shutil.copy2(path1 + v[2], file_name)
                shutil.copy2(path2 + v[3], file_name)


MkDir(data)
