docker stop new_serve
docker rm new_serve
docker rmi my_img
docker run -d --name serving_base tensorflow/serving:2.4.1

docker cp /Users/donruo/Desktop/project/bert_tasks/chinese_wwm_ext_L-12_H-768_A-12/serve/versions/ serving_base:/models/my_model

docker commit --change "ENV MODEL_NAME my_model" serving_base my_img
docker stop serving_base
docker rm serving_base

docker run --name new_serve -p 8501:8501 -p 8500:8500 my_img
