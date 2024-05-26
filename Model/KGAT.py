'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class KGAT(object):
    def __init__(self, data_config, pretrain_data, args):
        self._parse_args(data_config, pretrain_data, args)
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        self._build_inputs()

        """
        *********************************************************
        Create Model Parameters for CF & KGE parts.
        """
        self.weights = self._build_weights()

        """
        *********************************************************
        Compute Graph-based Representations of All Users & Items & KG Entities via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. bi: defined in 'KGAT: Knowledge Graph Attention Network for Recommendation', KDD2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. graphsage: defined in 'Inductive Representation Learning on Large Graphs', NeurIPS2017.
        """
        self._build_model_phase_I()
        """
        Optimize Recommendation (CF) Part via BPR Loss.
        """
        self._build_loss_phase_I()

        """
        *********************************************************
        Compute Knowledge Graph Embeddings via TransR.
        """
        self._build_model_phase_II()
        """
        Optimize KGE Part via BPR Loss.
        """
        self._build_loss_phase_II()

        self._statistics_params()

    def _parse_args(self, data_config, pretrain_data, args):
        # argument settings
        self.model_type = 'kgat'
        self.config = data_config
        self.pretrain_data = pretrain_data

        # 데이터 구성에 관련된 사용자, 아이템, 엔티티, 관계의 수
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        # K-Fold 교차 검증을 위한 fold 수
        # fold : 데이터를 나누는 단위, 100개로 나누겠다는 뜻
        self.n_fold = 100

        # 초기 Attentive Matrix A 설정
        # 사용자와 아이템 간의 상호작용을 나타냄
        self.A_in = data_config['A_in']

        # KGAT에서 사용되는 모든 헤드, 관계, 테일 및 값의 리스트를 설정
        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']

        # 인접 행렬의 유형을 설정
        self.adj_uni_type = args.adj_uni_type

        # 학습률 설정 learning rate의 약자인듯
        self.lr = args.lr

        # CF 부분의 임베딩 차원 설정
        self.emb_dim = args.embed_size
        # CF 부분의 배치 크기 설정
        self.batch_size = args.batch_size

        # KG 부분의 임베딩 차원 설정
        self.kge_dim = args.kge_size
        # KG 부분의 배치 크기 설정
        self.batch_size_kg = args.batch_size_kg
        
        # 레이어 크기 설정
        self.weight_size = eval(args.layer_size)
        # 레이어 수 설정
        self.n_layers = len(self.weight_size)

        # Graph Convolution Layer 유형 설정
        self.alg_type = args.alg_type
        # 모든 유형에 GCL의 유형과 수 추가
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)
        
        # 정규화 강도 설정
        self.regs = eval(args.regs)
        # 상세 출력 여부 설정
        self.verbose = args.verbose

    def _build_inputs(self):
        # placeholder definition
        # 플레이스 홀더 : 입력데이터를 나중에 제공하기 위해 사용되는 특별한 종류의 텐서
        # 실제로 실행할 때는 플레이스 홀더에 실제 데이터를 공급해 그래프를 실행함

        # 사용자, 선호 아이템, 비선호 아이템 플래이스 홀더
        self.users = tf.placeholder(tf.int32, shape=(None,))
        # 선호하는 아이템 플래이스 홀더
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        # 비선호 아이템 플래이스 홀더
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # for knowledge graph modeling (TransD)
        # TransD에서 사용되는 플레이스 홀더, 지식그래프 모델링 목적
        # Attentive Matrix A의 값을 받음
        self.A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)], name='A_values')

        # 지식그래프 모델링 목적
        # 지식그래프의 head, relation, tail을 나타냄
        # tail은 긍정적, 부정적 아이템을 구분함
        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

        # dropout: node dropout (adopted on the ego-networks);
        # message dropout (adopted on the convolution operations).

        # dropout : 신경망에서 사용되는 정규화기법 중 하나, 과적합을 방지함
        # 훈련중에 네트워크의 일부를 랜덤하게 제거해 모델이 다양한 부분을 학습하도록 강제함

        # 노드와 메세지 드롭아웃 플레이스 홀더
        # node dropout : 노드를 랜덤하게 비활성화, 노이즈에 강건하게 학습되도록 돕고 과적합을 방지
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        # message dropout : 메시지 전달 단계중에 일부 메시지를 랜덤하게 무시함
        # 노이즈에 강건하게되고, 훈련 데이터에서 잘못된 패턴을 학습하는 것을 방지
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

    def _build_weights(self):
        # 모든 가중치 저장
        all_weights = dict()

        # 가중치 초기화
        initializer = tf.contrib.layers.xavier_initializer()

        # 사전 훈련된 데이터가 없다면
        if self.pretrain_data is None:
            # user, entity 임베딩 초기화
            all_weights['user_embed'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embed')
            all_weights['entity_embed'] = tf.Variable(initializer([self.n_entities, self.emb_dim]), name='entity_embed')
            print('using xavier initialization')
        # 사전 훈련된 데이터가 있다면
        else:
            # 사전 훈련된 user, entity 임베딩 가져오기
            all_weights['user_embed'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                    name='user_embed', dtype=tf.float32)

            item_embed = self.pretrain_data['item_embed']
            other_embed = initializer([self.n_entities - self.n_items, self.emb_dim])
            all_weights['entity_embed'] = tf.Variable(initial_value=tf.concat([item_embed, other_embed], 0),
                                                      trainable=True, name='entity_embed', dtype=tf.float32)
            
            print('using pretrained initialization')
        # relation 임베딩 초기화
        all_weights['relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]),
                                                    name='relation_embed')
        # 관계별 변환 매트릭스 초기화
        all_weights['trans_W'] = tf.Variable(initializer([self.n_relations, self.emb_dim, self.kge_dim]))

        # 가중치 크기 저장 리스트
        # emb_dim : 임베딩 차원
        # weight_size : 숨겨진 레이어 크기
        # weight_size_list : [임베딩 차원, 첫 번째 숨겨진 레이어 크기, 두 번째 숨겨진 레이어 크기, ...]
        self.weight_size_list = [self.emb_dim] + self.weight_size


        # 가중치 : 인공 신경망에서 입력과 출력 사이의 관계를 나타내는 매개변수
        # 이런 가중치를 조정해 입력데이터와 원하는 출력 사이의 관계를 학습함

        # 편향(bias) : 뉴런이 활성화되는 정도를 조절하는 매개변수
        # 편향은 뉴런이 얼마나 쉽게 활성화 되는지를 결정하고, 입력의 가중합에 더해져 활성화 함수로 전달됨

        # 레이어 수 만큼 반복
        for k in range(self.n_layers):
            # k번째 GCN 레이어의 가중치와 편향을 나타냄
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)
            # k번째 BI-LSTM 레이어의 가중치와 편향을 나타냄
            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            # k번째 MLP 레이어의 가중치와 편향을 나타냄
            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([2 * self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights

    def _build_model_phase_I(self):
        # bi-interaction embedding 생성
        self.ua_embeddings, self.ea_embeddings = self._create_bi_interaction_embed()

        # 사용자, 긍정적 아이템, 부정적 아이템에 대한 임베딩 조회
        self.u_e = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_e = tf.nn.embedding_lookup(self.ea_embeddings, self.pos_items)
        self.neg_i_e = tf.nn.embedding_lookup(self.ea_embeddings, self.neg_items)

        # 배치 예측 계산
        self.batch_predictions = tf.matmul(self.u_e, self.pos_i_e, transpose_a=False, transpose_b=True)

    def _build_model_phase_II(self):
        # node, edge에 대한 embedding을 가져옴
        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e = self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)

        # TransE 점수 생성
        # 이 점수는 지식그래프의 head, relation, tail에 대한 점수로 나타냄
        self.A_kg_score = self._generate_transE_score(h=self.h, t=self.pos_t, r=self.r)

        # Attentive Matrix A 생성
        # 이 행렬은 지식 그래프에서 각 노드 간의 관계를 나타냄
        # 이 행렬을 사용해 사용자와 아이템 간 상호작용을 모델링
        self.A_out = self._create_attentive_A_out()

    # 지식그래프로부터 관계 정보를 추론해 사용자와 아이템간의 상호작용을 모델링
    def _get_kg_inference(self, h, r, pos_t, neg_t):
        # user, item embedding 결합
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        # 텐서 차원 확장, embeddings를 확장해 3차원으로 변환
        embeddings = tf.expand_dims(embeddings, 1)

        # head & tail entity embeddings: batch_size *1 * emb_dim
        # 사용자, 긍정적 아이템, 부정적 아이템, 관계에 대한 임베딩 가져옴
        # 각각의 임베딩은 차원을 확장해 3D 텐서로 변환
        h_e = tf.nn.embedding_lookup(embeddings, h)
        pos_t_e = tf.nn.embedding_lookup(embeddings, pos_t)
        neg_t_e = tf.nn.embedding_lookup(embeddings, neg_t)

        # relation embeddings: batch_size * kge_dim
        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)

        # 관계에 대한 가중치 적용
        # relation transform weights: batch_size * kge_dim * emb_dim
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        # batch_size * 1 * kge_dim -> batch_size * kge_dim
        # 관계에 대한 변환 매트릭스인 trans_M을 사용해 head와 tail 엔티티의 임베딩 변환
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
        pos_t_e = tf.reshape(tf.matmul(pos_t_e, trans_M), [-1, self.kge_dim])
        neg_t_e = tf.reshape(tf.matmul(neg_t_e, trans_M), [-1, self.kge_dim])
        
        # Remove the l2 normalization terms
        # h_e = tf.math.l2_normalize(h_e, axis=1)
        # r_e = tf.math.l2_normalize(r_e, axis=1)
        # pos_t_e = tf.math.l2_normalize(pos_t_e, axis=1)
        # neg_t_e = tf.math.l2_normalize(neg_t_e, axis=1)

        # 변환된 head, relation, tail 반환
        return h_e, r_e, pos_t_e, neg_t_e

    # 추천 시스템 모델의 학습 과정 중 손실 함수 정의 및 최적화를 위한 옵티마이저 설정
    def _build_loss_phase_I(self):
        # 긍정 아이템, 부정아이템과 사용자 임베딩간의 내적으로 계산해 점수 계산
        pos_scores = tf.reduce_sum(tf.multiply(self.u_e, self.pos_i_e), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_e, self.neg_i_e), axis=1)

        # 정규화 항 : 과적합을 방지하기 위해 사용됨
        # 모델의 복잡성을 제어하고, 가중치가 작은 값을 가지도록 유도해 일반화 성능을 향상시킴
        # L2 정규화 : 모델의 가중치의 제곱값을 손실함수에 추가, 이를 통해 가중치의 값이 너무 커지지 않도록 제한

        # 정규화 항 계산
        # L2 정규화를 적용한 사용자와 아이템 임베딩에 대한 손실 계산
        regularizer = tf.nn.l2_loss(self.u_e) + tf.nn.l2_loss(self.pos_i_e) + tf.nn.l2_loss(self.neg_i_e)
        regularizer = regularizer / self.batch_size

        # BPR(Bayesian Personalized Ranking) 손실함수를 사용해 긍정아이템과 부정아이템 간의 점수 차이를 최소화
        base_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        # base_loss = tf.negative(tf.reduce_mean(maxi))

        # 손실 함수 (Loss Function) : 모델의 예측 값과 실제 값 사이의 차이를 측정하는 함수
        # 딥러닝 모델은 이 손실 함수를 최소화 하도록 훈련됨

        # 최종 손실 함수 계산
        self.base_loss = base_loss
        self.kge_loss = tf.constant(0.0, tf.float32, [1])
        self.reg_loss = self.regs[0] * regularizer
        self.loss = self.base_loss + self.kge_loss + self.reg_loss

        # 옵티마이저 : 모델의 가중치를 업데이트하는 방법을 결정하는 알고리즘
        # 손실함수의 그래디언트를 사용해 모델의 매개변수를 조정해 손실을 최소화 하도록 함
        # 최적의 매개변수를 찾는것이 목적

        # 최적화를 위한 옵티마이저 설정
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _build_loss_phase_II(self):
        
        # head, relation, tail에 대한 점수 계산
        def _get_kg_score(h_e, r_e, t_e):
            kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
            return kg_score

        # 긍정적, 부정적 샘플에 대한 점수 계산
        pos_kg_score = _get_kg_score(self.h_e, self.r_e, self.pos_t_e)
        neg_kg_score = _get_kg_score(self.h_e, self.r_e, self.neg_t_e)
        
        # Knowledge Graph 손실 계산
        kg_loss = tf.reduce_mean(tf.nn.softplus(-(neg_kg_score - pos_kg_score)))
        # maxi = tf.log(tf.nn.sigmoid(neg_kg_score - pos_kg_score))
        # kg_loss = tf.negative(tf.reduce_mean(maxi))
        
        # 정규화 항 계싼
        # L2 정규화를 적용한 head, relation, tail에 대한 손실 계산
        kg_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + \
                      tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e)
        kg_reg_loss = kg_reg_loss / self.batch_size_kg

        # 최종 손실 함수 계산
        self.kge_loss2 = kg_loss
        self.reg_loss2 = self.regs[1] * kg_reg_loss
        self.loss2 = self.kge_loss2 + self.reg_loss2

        # 최적화를 위한 옵티마이저 설정
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)

    # Attention 메커니즘을 사용해 그래프에 있는 노드간의 상호작용을 포착하는 임베딩 생성
    def _create_bi_interaction_embed(self):
        # 그래프 인접 행렬 할당
        A = self.A_in
        # 입력 그래프의 인접성을 나타내는 A 행렬을 여러 부분 행렬로 분할
        # 메모리 효율성을 높이기 위함
        A_fold_hat = self._split_A_hat(A)

        # 초기 임베딩, 사용자와 엔티티의 임베딩을 결합한 것
        ego_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        # 초기 임베딩을 리스트에 추가, 이후에 각 레이어에서 생성된 임베딩도 추가될 예정
        all_embeddings = [ego_embeddings]

        # 각 레이어에 대한 반복
        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            temp_embed = []
            # 이웃 메시지 계산
            # 주어진 인접 행렬을 작은부분으로 분할한 후 각 부분 행렬을 사용해 이웃 메시지 계산
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings[:49702]))

            # 계산된 이웃 메시지를 모두 결합해 side_embeddings 생성
            # 현재 노드와 이웃 관계를 나타내는 메시지
            side_embeddings = tf.concat(temp_embed, 0)

            # 현재 노드의 임베딩과 이웃 노드의 메시지를 결합해 새로운 임베딩 생성
            add_embeddings = ego_embeddings[:49702] + side_embeddings

            # 1. add_embeddings를 선형 변환
            # 2. Leaky ReLU 활성화 함수를 적용해 이웃의 메시지를 변환
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(add_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # 이진 상호 작용 메시지 계싼
            # 현재 노드와 이웃 노드 간의 요소별 곱을 계산
            bi_embeddings = tf.multiply(ego_embeddings[:49702], side_embeddings)
            # 1. 이진 상호 작용 메시지를 선형 변환
            # 2. Leaky ReLU 활성화 함수를 적용해 변환
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # 이진 상호 작용 메시지와 이웃의 메시지를 결합해 새로운 임베딩 생성
            ego_embeddings = bi_embeddings + sum_embeddings
            # 메시지 드롭아웃을 적용해 임베딩 업데이트 (과적합 방지를 위해 일부 메시지 제거)
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            ego_embeddings = ego_embeddings
            # 생성된 임베딩을 정규화해 분산을 일정하게 유지, L2정규화 사용
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            
            # 각 레이어에서 생성된 임베딩을 all_embeddings 리스트에 추가
            # 이 임베딩은 후속 레이어의 입력으로 사용
            all_embeddings += [norm_embeddings]
            all_embeddings[0] = all_embeddings[0][:49702]
        # 모든 임베딩을 결합해 최종 임베딩 행렬 생성
        all_embeddings = tf.concat(all_embeddings, 1)

        # 최종 임베딩 행렬을 사용자와 엔티티 임베딩으로 분할
        # 이 임베딩들은 후속 모델에 사용됨
        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    # 행렬을 작은 부분 행렬로 분할
    def _split_A_hat(self, X):
        # 분할된 작은 부분 행렬이 저장될 리스트
        A_fold_hat = []

        # 전체 노드수 / 작은 부분 행렬 수
        # 하나의 작은 부분 행렬의 길이
        fold_len = (self.n_users + self.n_entities) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    # Scipy의 희소 행렬 (CSR포맷)을 TensorFlow의 희소 텐서로 변환
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    # 주어진 리스트 (head, tail, A.values)를 사용해 주어진 크기의 희소텐서 in.shape에 대해 소프트맥스 함수를 적용
    def _create_attentive_A_out(self):
        indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        A = tf.sparse.softmax(tf.SparseTensor(indices, self.A_values, self.A_in.shape))
        return A

    # TransE 모델을 사용해 head, tail, relation에 대한 점수 생성
    def _generate_transE_score(self, h, t, r):
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        embeddings = tf.expand_dims(embeddings, 1)

        # head, tail에 대한 임베딩 가져옴
        h_e = tf.nn.embedding_lookup(embeddings, h)
        t_e = tf.nn.embedding_lookup(embeddings, t)

        # relation에 대한 임베딩 가져옴
        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)

        # relation transform weights: batch_size * kge_dim * emb_dim
        # 관계에 대한 변환 가중치 가져옴
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        # batch_size * 1 * kge_dim -> batch_size * kge_dim
        # 각 요소에 대한 변환 가중치를 적용해 head, tail의 임베딩을 변환
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
        t_e = tf.reshape(tf.matmul(t_e, trans_M), [-1, self.kge_dim])

        # l2-normalize
        # h_e = tf.math.l2_normalize(h_e, axis=1)
        # r_e = tf.math.l2_normalize(r_e, axis=1)
        # t_e = tf.math.l2_normalize(t_e, axis=1)

        # head, tail, relation 임베딩을 조합해 TransE 모델에서 정의한 점수 계산
        kg_score = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), 1)

        return kg_score

    # 모델 파라미터 수 계산 후 출력
    def _statistics_params(self):
        # number of params
        total_parameters = 0
        # 가중치 딕셔너리에 있는 각 변수에 대해 반복
        for variable in self.weights.values():
            # 각 변수의 모양을 가져와 각 차원의 크기를 곱해 변수의 파라미터 수를 구함
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    ## 모델의 훈련 및 평가 과정 담당 함수
    # 피드 딕셔너리를 사용해 모델 훈련 후 훈련 결과 반환
    def train(self, sess, feed_dict):
        return sess.run([self.opt, self.loss, self.base_loss, self.kge_loss, self.reg_loss], feed_dict)

    # 피드 딕셔너리를 사용해 Attentive Matrix A를 훈련 후 훈련 결과 반환
    def train_A(self, sess, feed_dict):
        return sess.run([self.opt2, self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)

    # 피드 딕셔너리를 사용해 모델을 평가하고 평과 결과(배치 예측값) 반환
    def eval(self, sess, feed_dict):
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return batch_predictions

    def update_attentive_A(self, sess):
        fold_len = len(self.all_h_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            feed_dict = {
                self.h: self.all_h_list[start:end],
                self.r: self.all_r_list[start:end],
                self.pos_t: self.all_t_list[start:end]
            }
            A_kg_score = sess.run(self.A_kg_score, feed_dict=feed_dict)
            kg_score += list(A_kg_score)

        kg_score = np.array(kg_score)

        new_A = sess.run(self.A_out, feed_dict={self.A_values: kg_score})
        new_A_values = new_A.values
        new_A_indices = new_A.indices

        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.A_in = sp.coo_matrix((new_A_values, (rows, cols)), shape=(self.n_users + self.n_entities,
                                                                       self.n_users + self.n_entities))
        if self.alg_type in ['org', 'gcn']:
            self.A_in.setdiag(1.)
