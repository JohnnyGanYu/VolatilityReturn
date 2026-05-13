#!/bin/bash
# Package v6 submission (LGB-only plan)
# All files flat in archive root, no subdirectories

set -e

OUTPUT="submission_v6.tar.gz"
MODEL_DIR="models_v6_submit"  # 解压后的模型目录

echo "Packaging v6 submission..."

# 创建临时目录
rm -rf _pkg_tmp
mkdir _pkg_tmp

# 代码文件
cp factor.py _pkg_tmp/
cp predict.py _pkg_tmp/

# LGB 模型文件（从模型目录复制，去掉子目录前缀）
cp ${MODEL_DIR}/lgb_ret5_dataset*.txt.gz _pkg_tmp/
cp ${MODEL_DIR}/lgb_ret60_dataset*.txt.gz _pkg_tmp/
cp ${MODEL_DIR}/lgb_ret5_global.txt.gz _pkg_tmp/
cp ${MODEL_DIR}/lgb_ret60_global.txt.gz _pkg_tmp/

# 权重文件
cp ${MODEL_DIR}/ensemble_weights.json _pkg_tmp/

# 论文（如果有的话）
# cp C114.pdf _pkg_tmp/

# 打包
cd _pkg_tmp
tar -czf ../${OUTPUT} *
cd ..

# 检查
SIZE=$(du -sh ${OUTPUT} | cut -f1)
COUNT=$(tar -tzf ${OUTPUT} | wc -l)
echo ""
echo "Done: ${OUTPUT}"
echo "  Size: ${SIZE}"
echo "  Files: ${COUNT}"
echo ""
echo "Contents:"
tar -tzf ${OUTPUT} | head -10
echo "..."

# 清理
rm -rf _pkg_tmp
