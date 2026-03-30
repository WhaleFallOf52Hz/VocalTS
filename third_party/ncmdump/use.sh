# 处理单个或多个文件并输出到指定目录
ncmdump 1.ncm 2.ncm -o output_dir
ncmdump "linked_data/inference_data/0003/ヨルシカ - だから僕は音楽を辞めた.ncm" -o linked_data/inference_data/0002/

# 处理文件夹下的所有以 ncm 为扩展名并输出到指定目录，不包含子文件夹
ncmdump -d source_dir -o output_dir

# 递归处理文件夹并输出到指定目录，并保留目录结构
ncmdump -d source_dir -o output_dir -r