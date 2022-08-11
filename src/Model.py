import gurobipy as gp
from AlgorithmUserFairness import IndividualLossVariance


class UGF():
    
    def __init__(self, list_X_est): # recebe uma lista de 10 X_est
        self.list_X_est = list_X_est

    @staticmethod
    def _build_fairness_optimizer(list_X_est, list_X_est_Rindv, group_df_list):
        # Parâmetros do problema
        qtd_usuarios = []
        qtd_rindvs = []
        for i in range(3):
            qtd_usuarios[i] = len(group_df_list[i].index) # quantidade de usuários de um determinado grupo de usuários
            qtd_rindvs[i] = len(list_X_est[0].columns)   # quantidade de colunas da primeira X_est (mesmo para todos os grupos)

        i = 0
        mat_preferencias = []
        for m in range(3): # montar as três matrizes de preferências (três possíveis grupos de usuários)
            for u in range(len(qtd_usuarios)): # considerando as linhas de uma das matrizes igual ao número de usuários do grupo
                    if (u == )
                list_Rind_user[u] = list_X_est_Rindv

            mat_preferencias[i] = list_X_est[0]['Loss']
        
        for group in self.G: #G [user1, user2, user3, user4]
            for user in G[group]:


        # montar três matrizes de preferência
        # cada matriz de preferência terá x linha (número de usuários do grupo) e 10 colunas (número de Rindv calculadas para cada usuário)
        for X_est in list_X_est:



    @staticmethod
    def _format_result(model, df):
        """
        format the gurobi results to dataframe.
        :param model: optimized gurobi model
        :param df: the pandas dataframe to add the optimized results into
        :return: None
        """
        for v in model.getVars():
            v_s = v.varName.split('_')
            uid = int(v_s[0])
            iid = int(v_s[1])
            df.loc[(df['uid'] == uid) & (df['iid'] == iid), 'q'] = int(v.x)

    def _print_metrics(self, df, metrics, message='metric scores'):
        """
        Print out evaluation scores
        :param df: the dataframe contains the data for evaluation
        :param metrics: a list, contains the metrics to report
        :param message: a string, for print message
        :return: None
        """
        results = evaluation_methods(df, metrics=metrics)
        r_string = ""
        for i in range(len(metrics)):
            r_string = r_string + metrics[i] + "=" + '{:.4f}'.format(results[i]) + " "
        print(message + ": " + r_string)
        # write the message into the log file
        self.logger.info(message + ": " + r_string)

    def train(self):
        """
        Train fairness model
        """
        # model = read('gurobi_model.mps')

        # Prepare data
        all_df = self.data_loader.rank_df.copy(deep=True)    # the dataframe with entire test data
        self._check_df_format(all_df)   # check the dataframe format
        group_df_list = [self.data_loader.g1_df.copy(deep=True),
                         self.data_loader.g2_df.copy(deep=True)]  # group 1 (active), group 2 (inactive)

        # Print original evaluation results
        self.logger.info('Model:{} | Dataset:{} | Group:{} |  Epsilon={} | K={} | GRU_metric={}'
                         .format(self.model_name, self.dataset_name, self.group_name,
                                 self.epsilon, self.k, self.fairness_metric))
        self._print_metrics(all_df, self.eval_metric_list, 'Before optimization overall scores           ')
        self._print_metrics(group_df_list[0], self.eval_metric_list, 'Before optimization group 1 (active) scores  ')
        self._print_metrics(group_df_list[1], self.eval_metric_list, 'Before optimization group 2 (inactive) scores')

        # build optimizer
        m, var_score_list, metric_list = \
            self._build_fairness_optimizer(group_df_list, self.k, metric=self.fairness_metric, name='UGF_f1')

        # |group_1_recall - group_2_recall| <= epsilon
        m.addConstr(metric_list[0] - metric_list[1] <= self.epsilon)
        m.addConstr(metric_list[1] - metric_list[0] <= self.epsilon)

        # Set objective function
        m.setObjective(gp.quicksum(var_score_list), GRB.MAXIMIZE)

        # Optimize model
        m.optimize()

        # m.write('gurobi_model.mps')

        # Format the output results and update q column of the dataframe
        self._format_result(m, all_df)
        group_df_list[0].drop(columns=['q'], inplace=True)
        group_df_list[0] = pd.merge(group_df_list[0], all_df, on=['uid', 'iid', 'score', 'label'], how='left')
        group_df_list[1].drop(columns=['q'], inplace=True)
        group_df_list[1] = pd.merge(group_df_list[1], all_df, on=['uid', 'iid', 'score', 'label'], how='left')

        # Print updated evaluation results
        self._print_metrics(all_df, self.eval_metric_list, 'After optimization overall metric scores     ')
        self._print_metrics(group_df_list[0], self.eval_metric_list, 'After optimization group 1 (active) scores   ')
        self._print_metrics(group_df_list[1], self.eval_metric_list, 'After optimization group 2 (inactive) scores ')
        self.logger.info('\n\n')


if __name__ == '__main__':
    """
    Please update the following block for different datasets
    """
    ############### Parameters to be changed for different datasets ###########
    epsilon = 0.0                           # fairness constraint coefficient
    dataset_folder = '../dataset'           # dataset directory
    dataset_name = '5Beauty-rand'           # dataset name
    model_name = 'NCF'                      # model name (which model does this ranking file generated by)
    group_name_title = 'sum_0.05'           # grouping method name for distinguish different experiment results
    logger_dir = os.path.join('../results/', model_name)    # logging file path

    data_path = os.path.join(dataset_folder, dataset_name)
    rank_file = model_name + '_rank.csv'                                    # original input ranking csv file name
    group_1_file = group_name_title + '_price_active_test_ratings.txt'      # advantaged group testing file name
    group_2_file = group_name_title + '_price_inactive_test_ratings.txt'    # disadvantaged group testing file name
    ############################################################################


    #print(logger_dir)
    print('\nAntes do comando')
    if not os.path.exists(logger_dir):
        os.mkdir(logger_dir)
    #if not os.path.exists("../results/NCF"):
    #    os.mkdir('../results/NCF')

    print('\nDepois do comando')

    logger_file = model_name + '_' + dataset_name + '_' + group_name_title + '_reRank_result.log'
    logger_path = os.path.join(logger_dir, logger_file)
    dl = DataLoader(data_path, rank_file=rank_file, group_1_file=group_1_file, group_2_file=group_2_file)

    logger = create_logger(name='result_logger', path=logger_path)

    metrics = ['ndcg', 'f1']
    topK = ['10']

    metrics_list = [metric + '@' + k for metric in metrics for k in topK]

    UGF_model = UGF(dl, k=10, eval_metric_list=metrics_list, fairness_metric='f1',
                    epsilon=epsilon, logger=logger, model_name=model_name, group_name=group_name_title)
    UGF_model.train()


