export CUDA_VISIBLE_DEVICES=5
    python main_knn_kmeans_in+kmeans_load_nokmeans.py \
      -a resnet50 \
      --gpu 4 \
      --seed 14 \
      --batch-size 64 \
      --mlp --aug-plus \
      --data-A '../dataset/OfficeHomeDataset_10072016/Product' \
      --data-B '../dataset/OfficeHomeDataset_10072016/Clipart' \
      --num-cluster 65 \
      --epsilon 0.05 \
      --sinkhorn-iterations 100 \
      --temperature 0.2 \
      --exp-dir "officehome_product-clipart_inoload_ndomain_cross" \
      --lr 0.00025 \
      --epochs 200 \
      --topK 50 \
      --prec-nums '1,5,15' 
