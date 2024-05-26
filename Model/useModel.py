import tensorflow as tf
from KGAT import KGAT
from utility.parser import parse_args
from utility.helper import *
from utility.batch_test import *
from utility.load_data import *
from utility.parser import parse_args

from time import time

import numpy as np

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


config = dict()
# user, item, relation, entity 수
config['n_users'] = data_generator.n_users
config['n_items'] = data_generator.n_items
config['n_relations'] = data_generator.n_relations
config['n_entities'] = data_generator.n_entities

def load_pretrained_data(args):
    pretrain_path = 'D:/DEV/knowledge_graph_attention_network/pretrain/mykgat/mf.npz'
    try:
        pretrain_data = np.load(pretrain_path)
    except Exception:
        pretrain_data = None
    return pretrain_data

pretrain_data = load_pretrained_data(args)

config = dict()

# user, item, relation, entity 수
config['n_users'] = data_generator.n_users
config['n_items'] = data_generator.n_items
config['n_relations'] = data_generator.n_relations
config['n_entities'] = data_generator.n_entities

# laplacian 행렬 로드
# 모델의 그래프 구조를 정의하는 라플라시안 행렬 설정
config['A_in'] = sum(data_generator.lap_list)

# KG Triplets 로드
config['all_h_list'] = data_generator.all_h_list
config['all_r_list'] = data_generator.all_r_list
config['all_t_list'] = data_generator.all_t_list
config['all_v_list'] = data_generator.all_v_list

def generate_recommendations(sess, model, user_id, entity_embed, relation_embed):
    random_nums = np.random.choice(config['n_items'], size=50, replace=False)
    t1 = time()
    user_scores = []
    for item_id in random_nums:
        score = sess.run(model._generate_transE_score(user_id, item_id, 2))
        user_scores.append(score)
    
    # 점수를 정렬하여 상위 N개 아이템 추천
    top_k = 10  # 추천할 상위 N개 아이템
    user_scores = np.array(user_scores).flatten()  # 스코어를 1차원 배열로 변환
    for score in user_scores :
        print(score)
    sorted_item_indices = np.argsort(user_scores)[:top_k]
    recommendations = sorted_item_indices.tolist()
    t2 = time()
    print(t2 - t1)
    return recommendations

# 세션 시작
with tf.Session() as sess:
    # 세션 초기화 및 모델 복원
    # 모델 객체 생성
    model = KGAT(data_config=config, pretrain_data=pretrain_data, args=args)
    embed_size = 64
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "D:/DEV/knowledge_graph_attention_network/weights/mykgat/weights-9")

    htr = np.load("D:/DEV/knowledge_graph_attention_network/pretrain/mykgat/htr.npz")

    user_embed = model.pretrain_data['user_embed'][0]
    entity_embed = model.pretrain_data['item_embed'][0]
    relation_embed = htr['relation_embed'][0]

    # 사용자, 관계, 아이템의 데이터를 입력으로 사용하여 예측
    # predictions = model._generate_transE_score(3, 511, 2)
    recommendations = generate_recommendations(sess, model, 12, entity_embed, relation_embed)

    for i in recommendations :
        print(i)