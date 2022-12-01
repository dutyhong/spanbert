# 作者 ：duty
# 创建时间 ：2022/9/19 11:19 上午
# 文件 ：sample_process.py

## 将每次新增的样本进行处理和历史样本合一起
def sample_combine(old_sample_path,increase_samples):
	file = open(old_sample_path, "r")
	file2 = open("./data/latest_version_samples", "w")
	file3 = open("./data/new_optimize_samples", "w")
	for line in file.readlines():
		file2.write(line)
	for line in increase_samples:
		file2.write(line+"\n")
		file3.write(line+"\n")
	file2.close()
	file.close()
	file3.close()
	# 追加新增样本到历史样本里
	file4 = open(old_sample_path, "a")
	for line in increase_samples:
		file4.write(line+"\n")
	file4.close()

def input_data_trans(df):
	samples = []
	for index, row in df.iterrows():
		# print(row)
		sent = row[0]
		tags = row[1]
		print(sent)
		print(tags)
		samples.append(sent+";"+tags)
	return samples