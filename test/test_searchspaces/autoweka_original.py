from  hyperopt import hp
import math

space = (
    hp.choice('auto_param_2', [
        {'attributesearch':'NONE'},
        (
            hp.choice('auto_param_4', [
                {
                    'attributesearch':'weka.attributeSelection.BestFirst',
                    'assearch_wasbf__00__01_0_D':hp.choice('assearch_wasbf__00__01_0_D', ['2', '1', '0', ]),
                    'assearch_wasbf__00__02_1_INT_N':hp.quniform('assearch_wasbf__00__02_1_INT_N', 2.0, 10.0,1.0),
                    'assearch_wasbf__00__03_2_S':hp.choice('assearch_wasbf__00__03_2_S', ['0', ]),
                },
                {
                    'attributesearch':'weka.attributeSelection.GreedyStepwise',
                    'assearch_wasgs__01__01_0_C':hp.choice('assearch_wasgs__01__01_0_C', ['REMOVE_PREV', 'REMOVED', ]),
                    'assearch_wasgs__01__02_1_B':hp.choice('assearch_wasgs__01__02_1_B', ['REMOVE_PREV', 'REMOVED', ]),
                    'assearch_wasgs__01__03_auto_param_7':hp.choice('assearch_wasgs__01__03_auto_param_7', [
                        {
                            'assearch_wasgs__01__03__00__00_2_R':'REMOVED',
                            'assearch_wasgs__01__03__00__01_3_T':hp.uniform('assearch_wasgs__01__03__00__01_3_T', 0.0, 20.0),
                        },
                        {
                            'assearch_wasgs__01__03__01__00_2_R':'REMOVE_PREV',
                            'assearch_wasgs__01__03__01__01_4_INT_N':hp.qloguniform('assearch_wasgs__01__03__01__01_4_INT_N', math.log(10.0), math.log(1000.0), 1.0),
                        },
                    ]),
                },
                {
                    'attributesearch':'weka.attributeSelection.Ranker',
                    'assearch_wasr__02__01_0_T':hp.uniform('assearch_wasr__02__01_0_T', 0.2, 10.0),
                },
            ]),
            hp.choice('auto_param_11', [
                {
                    'attributeeval':'weka.attributeSelection.CfsSubsetEval',
                    'aseval_wascse__00__01_0_M':hp.choice('aseval_wascse__00__01_0_M', ['REMOVE_PREV', 'REMOVED', ]),
                    'aseval_wascse__00__02_1_L':hp.choice('aseval_wascse__00__02_1_L', ['REMOVE_PREV', 'REMOVED', ]),
                },
                {
                    'attributeeval':'weka.attributeSelection.CorrelationAttributeEval',
                },
                {
                    'attributeeval':'weka.attributeSelection.GainRatioAttributeEval',
                },
                {
                    'attributeeval':'weka.attributeSelection.InfoGainAttributeEval',
                    'aseval_wasigae__03__01_0_M':hp.choice('aseval_wasigae__03__01_0_M', ['REMOVE_PREV', 'REMOVED', ]),
                    'aseval_wasigae__03__02_1_B':hp.choice('aseval_wasigae__03__02_1_B', ['REMOVE_PREV', 'REMOVED', ]),
                },
                {
                    'attributeeval':'weka.attributeSelection.OneRAttributeEval',
                    'aseval_wasorae__04__01_0_S':hp.choice('aseval_wasorae__04__01_0_S', ['0', ]),
                    'aseval_wasorae__04__02_1_F':hp.quniform('aseval_wasorae__04__02_1_F', 2.0, 15.0,1.0),
                    'aseval_wasorae__04__03_2_D':hp.choice('aseval_wasorae__04__03_2_D', ['REMOVE_PREV', 'REMOVED', ]),
                    'aseval_wasorae__04__04_3_INT_B':hp.qloguniform('aseval_wasorae__04__04_3_INT_B', math.log(1.0), math.log(64.0), 1.0),
                },
                {
                    'attributeeval':'weka.attributeSelection.PrincipalComponents',
                    'aseval_waspc__05__01_1_C':hp.choice('aseval_waspc__05__01_1_C', ['REMOVE_PREV', 'REMOVED', ]),
                    'aseval_waspc__05__02_2_R':hp.uniform('aseval_waspc__05__02_2_R', 0.5, 1.0),
                    'aseval_waspc__05__03_3_O':hp.choice('aseval_waspc__05__03_3_O', ['REMOVE_PREV', 'REMOVED', ]),
                    'aseval_waspc__05__04_auto_param_18':hp.choice('aseval_waspc__05__04_auto_param_18', [
                        {
                            'aseval_waspc__05__04__00__00_num_HIDDEN':'0',
                            'aseval_waspc__05__04__00__01_1_INT_A':hp.choice('aseval_waspc__05__04__00__01_1_INT_A', ['-1', ]),
                        },
                        {
                            'aseval_waspc__05__04__01__00_num_HIDDEN':'1',
                            'aseval_waspc__05__04__01__01_2_INT_A':hp.qloguniform('aseval_waspc__05__04__01__01_2_INT_A', math.log(1.0), math.log(1024.0), 1.0),
                        },
                    ]),
                },
                {
                    'attributeeval':'weka.attributeSelection.ReliefFAttributeEval',
                    'aseval_wasrfae__06__01_0_INT_K':hp.qloguniform('aseval_wasrfae__06__01_0_INT_K', math.log(2.0), math.log(64.0), 1.0),
                    'aseval_wasrfae__06__02_auto_param_22':hp.choice('aseval_wasrfae__06__02_auto_param_22', [
                        {
                            'aseval_wasrfae__06__02__00__00_1_W':'REMOVED',
                            'aseval_wasrfae__06__02__00__01_2_INT_A':hp.qloguniform('aseval_wasrfae__06__02__00__01_2_INT_A', math.log(1.0), math.log(8.0), 1.0),
                        },
                        {
                            'aseval_wasrfae__06__02__01__00_1_W':'REMOVE_PREV',
                        },
                    ]),
                },
                {
                    'attributeeval':'weka.attributeSelection.SymmetricalUncertAttributeEval',
                    'aseval_wassuae__07__01_0_M':hp.choice('aseval_wassuae__07__01_0_M', ['REMOVE_PREV', 'REMOVED', ]),
                },
            ]),
            {'attributetime':'900.0'},
        ),
    ]),
    hp.choice('is_base', [
        hp.choice('base_choice', [
            {
                'targetclass':'weka.classifiers.bayes.BayesNet',
                '_0_wcbbn__00__01_D':hp.choice('_0_wcbbn__00__01_D', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcbbn__00__02_Q':hp.choice('_0_wcbbn__00__02_Q', ['weka.classifiers.bayes.net.search.local.TAN', 'weka.classifiers.bayes.net.search.local.HillClimber', 'weka.classifiers.bayes.net.search.local.LAGDHillClimber', 'weka.classifiers.bayes.net.search.local.SimulatedAnnealing', 'weka.classifiers.bayes.net.search.local.TabuSearch', 'weka.classifiers.bayes.net.search.local.K2', ]),
            },
            {
                'targetclass':'weka.classifiers.bayes.NaiveBayes',
                '_0_wcbnb__01__01_K':hp.choice('_0_wcbnb__01__01_K', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcbnb__01__02_D':hp.choice('_0_wcbnb__01__02_D', ['REMOVE_PREV', 'REMOVED', ]),
            },
            {
                'targetclass':'weka.classifiers.bayes.NaiveBayesMultinomial',
            },
            {
                'targetclass':'weka.classifiers.functions.Logistic',
                '_0_wcfl__03__01_R':hp.loguniform('_0_wcfl__03__01_R', math.log(1.0E-12), math.log(10.0)),
            },
            {
                'targetclass':'weka.classifiers.functions.MultilayerPerceptron',
                '_0_wcfmp__04__01_L':hp.uniform('_0_wcfmp__04__01_L', 0.1, 1.0),
                '_0_wcfmp__04__02_M':hp.uniform('_0_wcfmp__04__02_M', 0.1, 1.0),
                '_0_wcfmp__04__03_B':hp.choice('_0_wcfmp__04__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcfmp__04__04_H':hp.choice('_0_wcfmp__04__04_H', ['t', 'i', 'o', 'a', ]),
                '_0_wcfmp__04__05_C':hp.choice('_0_wcfmp__04__05_C', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcfmp__04__06_R':hp.choice('_0_wcfmp__04__06_R', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcfmp__04__07_D':hp.choice('_0_wcfmp__04__07_D', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcfmp__04__08_S':hp.choice('_0_wcfmp__04__08_S', ['1', ]),
            },
            {
                'targetclass':'weka.classifiers.functions.SGD',
                '_0_wcfsgd__05__01_F':hp.choice('_0_wcfsgd__05__01_F', ['2', '1', '0', ]),
                '_0_wcfsgd__05__02_L':hp.loguniform('_0_wcfsgd__05__02_L', math.log(1.0E-5), math.log(0.1)),
                '_0_wcfsgd__05__03_R':hp.loguniform('_0_wcfsgd__05__03_R', math.log(1.0E-12), math.log(10.0)),
                '_0_wcfsgd__05__04_N':hp.choice('_0_wcfsgd__05__04_N', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcfsgd__05__05_M':hp.choice('_0_wcfsgd__05__05_M', ['REMOVE_PREV', 'REMOVED', ]),
            },
            {
                'targetclass':'weka.classifiers.functions.SMO',
                '_0_wcfsmo__06__01_0_C':hp.uniform('_0_wcfsmo__06__01_0_C', 0.5, 1.5),
                '_0_wcfsmo__06__02_1_N':hp.choice('_0_wcfsmo__06__02_1_N', ['2', '1', '0', ]),
                '_0_wcfsmo__06__03_2_M':hp.choice('_0_wcfsmo__06__03_2_M', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcfsmo__06__04_5_QUOTE_END':hp.choice('_0_wcfsmo__06__04_5_QUOTE_END', ['REMOVED', ]),
                '_0_wcfsmo__06__05_auto_param_33':hp.choice('_0_wcfsmo__06__05_auto_param_33', [
                    {
                        '_0_wcfsmo__06__05__00__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.NormalizedPolyKernel',
                        '_0_wcfsmo__06__05__00__01_4_npoly_E':hp.uniform('_0_wcfsmo__06__05__00__01_4_npoly_E', 0.2, 5.0),
                        '_0_wcfsmo__06__05__00__02_4_npoly_L':hp.choice('_0_wcfsmo__06__05__00__02_4_npoly_L', ['REMOVE_PREV', 'REMOVED', ]),
                    },
                    {
                        '_0_wcfsmo__06__05__01__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.PolyKernel',
                        '_0_wcfsmo__06__05__01__01_4_poly_E':hp.uniform('_0_wcfsmo__06__05__01__01_4_poly_E', 0.2, 5.0),
                        '_0_wcfsmo__06__05__01__02_4_poly_L':hp.choice('_0_wcfsmo__06__05__01__02_4_poly_L', ['REMOVE_PREV', 'REMOVED', ]),
                    },
                    {
                        '_0_wcfsmo__06__05__02__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.Puk',
                        '_0_wcfsmo__06__05__02__01_4_puk_S':hp.uniform('_0_wcfsmo__06__05__02__01_4_puk_S', 0.1, 10.0),
                        '_0_wcfsmo__06__05__02__02_4_puk_O':hp.uniform('_0_wcfsmo__06__05__02__02_4_puk_O', 0.1, 1.0),
                    },
                    {
                        '_0_wcfsmo__06__05__03__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.RBFKernel',
                        '_0_wcfsmo__06__05__03__01_4_rbf_G':hp.loguniform('_0_wcfsmo__06__05__03__01_4_rbf_G', math.log(1.0E-4), math.log(1.0)),
                    },
                ]),
            },
            {
                'targetclass':'weka.classifiers.functions.SimpleLogistic',
                '_0_wcfsl__07__01_S':hp.choice('_0_wcfsl__07__01_S', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcfsl__07__02_A':hp.choice('_0_wcfsl__07__02_A', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcfsl__07__03_auto_param_39':hp.choice('_0_wcfsl__07__03_auto_param_39', [
                    {
                        '_0_wcfsl__07__03__00__00_W_HIDDEN':'0',
                        '_0_wcfsl__07__03__00__01_1_W':hp.choice('_0_wcfsl__07__03__00__01_1_W', ['0', ]),
                    },
                    {
                        '_0_wcfsl__07__03__01__00_W_HIDDEN':'1',
                        '_0_wcfsl__07__03__01__01_2_W':hp.uniform('_0_wcfsl__07__03__01__01_2_W', 0.0, 1.0),
                    },
                ]),
            },
            {
                'targetclass':'weka.classifiers.functions.VotedPerceptron',
                '_0_wcfvp__08__01_INT_I':hp.quniform('_0_wcfvp__08__01_INT_I', 1.0, 10.0,1.0),
                '_0_wcfvp__08__02_INT_M':hp.qloguniform('_0_wcfvp__08__02_INT_M', math.log(5000.0), math.log(50000.0), 1.0),
                '_0_wcfvp__08__03_E':hp.uniform('_0_wcfvp__08__03_E', 0.2, 5.0),
            },
            {
                'targetclass':'weka.classifiers.lazy.IBk',
                '_0_wclib__09__01_E':hp.choice('_0_wclib__09__01_E', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wclib__09__02_INT_K':hp.qloguniform('_0_wclib__09__02_INT_K', math.log(1.0), math.log(64.0), 1.0),
                '_0_wclib__09__03_X':hp.choice('_0_wclib__09__03_X', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wclib__09__04_F':hp.choice('_0_wclib__09__04_F', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wclib__09__05_I':hp.choice('_0_wclib__09__05_I', ['REMOVE_PREV', 'REMOVED', ]),
            },
            {
                'targetclass':'weka.classifiers.lazy.KStar',
                '_0_wclks__10__01_INT_B':hp.quniform('_0_wclks__10__01_INT_B', 1.0, 100.0,1.0),
                '_0_wclks__10__02_E':hp.choice('_0_wclks__10__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wclks__10__03_M':hp.choice('_0_wclks__10__03_M', ['n', 'd', 'm', 'a', ]),
            },
            {
                'targetclass':'weka.classifiers.rules.DecisionTable',
                '_0_wcrdt__11__01_E':hp.choice('_0_wcrdt__11__01_E', ['auc', 'rmse', 'mae', 'acc', ]),
                '_0_wcrdt__11__02_I':hp.choice('_0_wcrdt__11__02_I', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcrdt__11__03_S':hp.choice('_0_wcrdt__11__03_S', ['weka.attributeSelection.Ranker', 'weka.attributeSelection.GreedyStepwise', 'weka.attributeSelection.BestFirst', ]),
                '_0_wcrdt__11__04_X':hp.choice('_0_wcrdt__11__04_X', ['4', '2', '3', '1', ]),
            },
            {
                'targetclass':'weka.classifiers.rules.JRip',
                '_0_wcrjr__12__01_N':hp.uniform('_0_wcrjr__12__01_N', 1.0, 5.0),
                '_0_wcrjr__12__02_E':hp.choice('_0_wcrjr__12__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcrjr__12__03_P':hp.choice('_0_wcrjr__12__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcrjr__12__04_INT_O':hp.quniform('_0_wcrjr__12__04_INT_O', 1.0, 5.0,1.0),
            },
            {
                'targetclass':'weka.classifiers.rules.OneR',
                '_0_wcror__13__01_INT_B':hp.qloguniform('_0_wcror__13__01_INT_B', math.log(1.0), math.log(32.0), 1.0),
            },
            {
                'targetclass':'weka.classifiers.rules.PART',
                '_0_wcrpart__14__01_INT_N':hp.quniform('_0_wcrpart__14__01_INT_N', 2.0, 5.0,1.0),
                '_0_wcrpart__14__02_INT_M':hp.qloguniform('_0_wcrpart__14__02_INT_M', math.log(1.0), math.log(64.0), 1.0),
                '_0_wcrpart__14__03_R':hp.choice('_0_wcrpart__14__03_R', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wcrpart__14__04_B':hp.choice('_0_wcrpart__14__04_B', ['REMOVE_PREV', 'REMOVED', ]),
            },
            {
                'targetclass':'weka.classifiers.rules.ZeroR',
            },
            {
                'targetclass':'weka.classifiers.trees.DecisionStump',
            },
            {
                'targetclass':'weka.classifiers.trees.J48',
                '_0_wctj__17__01_O':hp.choice('_0_wctj__17__01_O', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctj__17__02_U':hp.choice('_0_wctj__17__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctj__17__03_B':hp.choice('_0_wctj__17__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctj__17__04_J':hp.choice('_0_wctj__17__04_J', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctj__17__05_A':hp.choice('_0_wctj__17__05_A', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctj__17__06_S':hp.choice('_0_wctj__17__06_S', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctj__17__07_INT_M':hp.qloguniform('_0_wctj__17__07_INT_M', math.log(1.0), math.log(64.0), 1.0),
                '_0_wctj__17__08_C':hp.uniform('_0_wctj__17__08_C', 0.0, 1.0),
            },
            {
                'targetclass':'weka.classifiers.trees.LMT',
                '_0_wctlmt__18__01_B':hp.choice('_0_wctlmt__18__01_B', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctlmt__18__02_R':hp.choice('_0_wctlmt__18__02_R', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctlmt__18__03_C':hp.choice('_0_wctlmt__18__03_C', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctlmt__18__04_P':hp.choice('_0_wctlmt__18__04_P', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctlmt__18__05_INT_M':hp.qloguniform('_0_wctlmt__18__05_INT_M', math.log(1.0), math.log(64.0), 1.0),
                '_0_wctlmt__18__06_A':hp.choice('_0_wctlmt__18__06_A', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctlmt__18__07_auto_param_53':hp.choice('_0_wctlmt__18__07_auto_param_53', [
                    {
                        '_0_wctlmt__18__07__00__00_W_HIDDEN':'0',
                        '_0_wctlmt__18__07__00__01_1_W':hp.choice('_0_wctlmt__18__07__00__01_1_W', ['0', ]),
                    },
                    {
                        '_0_wctlmt__18__07__01__00_W_HIDDEN':'1',
                        '_0_wctlmt__18__07__01__01_2_W':hp.uniform('_0_wctlmt__18__07__01__01_2_W', 0.0, 1.0),
                    },
                ]),
            },
            {
                'targetclass':'weka.classifiers.trees.REPTree',
                '_0_wctrept__19__01_INT_M':hp.qloguniform('_0_wctrept__19__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                '_0_wctrept__19__02_V':hp.loguniform('_0_wctrept__19__02_V', math.log(1.0E-5), math.log(0.1)),
                '_0_wctrept__19__03_P':hp.choice('_0_wctrept__19__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctrept__19__04_auto_param_57':hp.choice('_0_wctrept__19__04_auto_param_57', [
                    {
                        '_0_wctrept__19__04__00__00_depth_HIDDEN':'0',
                        '_0_wctrept__19__04__00__01_1_INT_L':hp.choice('_0_wctrept__19__04__00__01_1_INT_L', ['-1', ]),
                    },
                    {
                        '_0_wctrept__19__04__01__00_depth_HIDDEN':'1',
                        '_0_wctrept__19__04__01__01_2_INT_L':hp.quniform('_0_wctrept__19__04__01__01_2_INT_L', 2.0, 20.0,1.0),
                    },
                ]),
            },
            {
                'targetclass':'weka.classifiers.trees.RandomForest',
                '_0_wctrf__20__01_INT_I':hp.qloguniform('_0_wctrf__20__01_INT_I', math.log(2.0), math.log(256.0), 1.0),
                '_0_wctrf__20__02_auto_param_61':hp.choice('_0_wctrf__20__02_auto_param_61', [
                    {
                        '_0_wctrf__20__02__00__00_features_HIDDEN':'0',
                        '_0_wctrf__20__02__00__01_1_INT_K':hp.choice('_0_wctrf__20__02__00__01_1_INT_K', ['1', ]),
                    },
                    {
                        '_0_wctrf__20__02__01__00_features_HIDDEN':'1',
                        '_0_wctrf__20__02__01__01_2_INT_K':hp.qloguniform('_0_wctrf__20__02__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                    },
                ]),
                '_0_wctrf__20__03_auto_param_64':hp.choice('_0_wctrf__20__03_auto_param_64', [
                    {
                        '_0_wctrf__20__03__00__00_depth_HIDDEN':'0',
                        '_0_wctrf__20__03__00__01_1_INT_depth':hp.choice('_0_wctrf__20__03__00__01_1_INT_depth', ['1', ]),
                    },
                    {
                        '_0_wctrf__20__03__01__00_depth_HIDDEN':'1',
                        '_0_wctrf__20__03__01__01_2_INT_depth':hp.quniform('_0_wctrf__20__03__01__01_2_INT_depth', 2.0, 20.0,1.0),
                    },
                ]),
            },
            {
                'targetclass':'weka.classifiers.trees.RandomTree',
                '_0_wctrt__21__01_INT_M':hp.qloguniform('_0_wctrt__21__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                '_0_wctrt__21__02_U':hp.choice('_0_wctrt__21__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                '_0_wctrt__21__03_auto_param_68':hp.choice('_0_wctrt__21__03_auto_param_68', [
                    {
                        '_0_wctrt__21__03__00__00_features_HIDDEN':'0',
                        '_0_wctrt__21__03__00__01_1_INT_K':hp.choice('_0_wctrt__21__03__00__01_1_INT_K', ['0', ]),
                    },
                    {
                        '_0_wctrt__21__03__01__00_features_HIDDEN':'1',
                        '_0_wctrt__21__03__01__01_2_INT_K':hp.qloguniform('_0_wctrt__21__03__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                    },
                ]),
                '_0_wctrt__21__04_auto_param_71':hp.choice('_0_wctrt__21__04_auto_param_71', [
                    {
                        '_0_wctrt__21__04__00__00_depth_HIDDEN':'0',
                        '_0_wctrt__21__04__00__01_1_INT_depth':hp.choice('_0_wctrt__21__04__00__01_1_INT_depth', ['0', ]),
                    },
                    {
                        '_0_wctrt__21__04__01__00_depth_HIDDEN':'1',
                        '_0_wctrt__21__04__01__01_2_INT_depth':hp.quniform('_0_wctrt__21__04__01__01_2_INT_depth', 2.0, 20.0,1.0),
                    },
                ]),
                '_0_wctrt__21__05_auto_param_74':hp.choice('_0_wctrt__21__05_auto_param_74', [
                    {
                        '_0_wctrt__21__05__00__00_back_HIDDEN':'0',
                        '_0_wctrt__21__05__00__01_1_INT_N':hp.choice('_0_wctrt__21__05__00__01_1_INT_N', ['0', ]),
                    },
                    {
                        '_0_wctrt__21__05__01__00_back_HIDDEN':'1',
                        '_0_wctrt__21__05__01__01_2_INT_N':hp.quniform('_0_wctrt__21__05__01__01_2_INT_N', 2.0, 5.0,1.0),
                    },
                ]),
            },
        ]),
        hp.choice('is_meta', [
            (
                hp.choice('meta_choice', [
                    {
                        'targetclass':'weka.classifiers.lazy.LWL',
                        '_0_wcllwl__00__01_K':hp.choice('_0_wcllwl__00__01_K', ['120', '10', '30', '60', '90', '-1', ]),
                        '_0_wcllwl__00__02_U':hp.choice('_0_wcllwl__00__02_U', ['4', '1', '2', '3', '0', ]),
                        '_0_wcllwl__00__03_A':hp.choice('_0_wcllwl__00__03_A', ['weka.core.neighboursearch.LinearNNSearch', ]),
                    },
                    {
                        'targetclass':'weka.classifiers.meta.AdaBoostM1',
                        '_0_wcmabm__01__01_INT_I':hp.qloguniform('_0_wcmabm__01__01_INT_I', math.log(2.0), math.log(128.0), 1.0),
                        '_0_wcmabm__01__02_Q':hp.choice('_0_wcmabm__01__02_Q', ['REMOVE_PREV', 'REMOVED', ]),
                        '_0_wcmabm__01__03_S':hp.choice('_0_wcmabm__01__03_S', ['1', ]),
                        '_0_wcmabm__01__04_auto_param_80':hp.choice('_0_wcmabm__01__04_auto_param_80', [
                            {
                                '_0_wcmabm__01__04__00__00_p_HIDDEN':'0',
                                '_0_wcmabm__01__04__00__01_1_P':hp.choice('_0_wcmabm__01__04__00__01_1_P', ['100', ]),
                            },
                            {
                                '_0_wcmabm__01__04__01__00_p_HIDDEN':'1',
                                '_0_wcmabm__01__04__01__01_2_INT_P':hp.quniform('_0_wcmabm__01__04__01__01_2_INT_P', 50.0, 100.0,1.0),
                            },
                        ]),
                    },
                    {
                        'targetclass':'weka.classifiers.meta.AttributeSelectedClassifier',
                        '_0_wcmasc__02__01_S':hp.choice('_0_wcmasc__02__01_S', ['weka.attributeSelection.Ranker', 'weka.attributeSelection.GreedyStepwise', 'weka.attributeSelection.BestFirst', ]),
                        '_0_wcmasc__02__02_E':hp.choice('_0_wcmasc__02__02_E', ['weka.attributeSelection.GainRatioAttributeEval', 'weka.attributeSelection.WrapperSubsetEval', 'weka.attributeSelection.OneRAttributeEval', 'weka.attributeSelection.InfoGainAttributeEval', 'weka.attributeSelection.HoldOutSubsetEvaluator', 'weka.attributeSelection.CfsSubsetEval', ]),
                    },
                    {
                        'targetclass':'weka.classifiers.meta.Bagging',
                        '_0_wcmb__03__01_INT_P':hp.quniform('_0_wcmb__03__01_INT_P', 10.0, 200.0,1.0),
                        '_0_wcmb__03__02_INT_I':hp.qloguniform('_0_wcmb__03__02_INT_I', math.log(2.0), math.log(128.0), 1.0),
                        '_0_wcmb__03__03_S':hp.choice('_0_wcmb__03__03_S', ['1', ]),
                        '_0_wcmb__03__04_O':hp.choice('_0_wcmb__03__04_O', ['REMOVE_PREV', 'REMOVED', ]),
                    },
                    {
                        'targetclass':'weka.classifiers.meta.ClassificationViaRegression',
                    },
                    {
                        'targetclass':'weka.classifiers.meta.LogitBoost',
                        '_0_wcmlb__05__01_INT_I':hp.qloguniform('_0_wcmlb__05__01_INT_I', math.log(2.0), math.log(128.0), 1.0),
                        '_0_wcmlb__05__02_INT_R':hp.quniform('_0_wcmlb__05__02_INT_R', 1.0, 5.0,1.0),
                        '_0_wcmlb__05__03_Q':hp.choice('_0_wcmlb__05__03_Q', ['REMOVE_PREV', 'REMOVED', ]),
                        '_0_wcmlb__05__04_L':hp.choice('_0_wcmlb__05__04_L', ['1e50', ]),
                        '_0_wcmlb__05__05_S':hp.choice('_0_wcmlb__05__05_S', ['1', ]),
                        '_0_wcmlb__05__06_auto_param_87':hp.choice('_0_wcmlb__05__06_auto_param_87', [
                            {
                                '_0_wcmlb__05__06__00__00_h_HIDDEN':'0',
                                '_0_wcmlb__05__06__00__01_1_H':hp.choice('_0_wcmlb__05__06__00__01_1_H', ['1', ]),
                            },
                            {
                                '_0_wcmlb__05__06__01__00_h_HIDDEN':'1',
                                '_0_wcmlb__05__06__01__01_2_H':hp.uniform('_0_wcmlb__05__06__01__01_2_H', 0.0, 1.0),
                            },
                        ]),
                        '_0_wcmlb__05__07_auto_param_90':hp.choice('_0_wcmlb__05__07_auto_param_90', [
                            {
                                '_0_wcmlb__05__07__00__00_f_HIDDEN':'0',
                                '_0_wcmlb__05__07__00__01_1_F':hp.choice('_0_wcmlb__05__07__00__01_1_F', ['0', ]),
                            },
                            {
                                '_0_wcmlb__05__07__01__00_f_HIDDEN':'1',
                                '_0_wcmlb__05__07__01__01_2_INT_F':hp.quniform('_0_wcmlb__05__07__01__01_2_INT_F', 1.0, 5.0,1.0),
                            },
                        ]),
                        '_0_wcmlb__05__08_auto_param_93':hp.choice('_0_wcmlb__05__08_auto_param_93', [
                            {
                                '_0_wcmlb__05__08__00__00_p_HIDDEN':'0',
                                '_0_wcmlb__05__08__00__01_1_P':hp.choice('_0_wcmlb__05__08__00__01_1_P', ['100', ]),
                            },
                            {
                                '_0_wcmlb__05__08__01__00_p_HIDDEN':'1',
                                '_0_wcmlb__05__08__01__01_2_INT_P':hp.quniform('_0_wcmlb__05__08__01__01_2_INT_P', 50.0, 100.0,1.0),
                            },
                        ]),
                    },
                    {
                        'targetclass':'weka.classifiers.meta.MultiClassClassifier',
                        '_0_wcmmcc__06__01_M':hp.choice('_0_wcmmcc__06__01_M', ['3', '1', '2', '0', ]),
                        '_0_wcmmcc__06__02_R':hp.uniform('_0_wcmmcc__06__02_R', 0.5, 4.0),
                        '_0_wcmmcc__06__03_P':hp.choice('_0_wcmmcc__06__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                        '_0_wcmmcc__06__04_S':hp.choice('_0_wcmmcc__06__04_S', ['1', ]),
                    },
                    {
                        'targetclass':'weka.classifiers.meta.RandomCommittee',
                        '_0_wcmrc__07__01_INT_I':hp.qloguniform('_0_wcmrc__07__01_INT_I', math.log(2.0), math.log(64.0), 1.0),
                        '_0_wcmrc__07__02_S':hp.choice('_0_wcmrc__07__02_S', ['1', ]),
                    },
                    {
                        'targetclass':'weka.classifiers.meta.RandomSubSpace',
                        '_0_wcmrss__08__01_INT_I':hp.qloguniform('_0_wcmrss__08__01_INT_I', math.log(2.0), math.log(64.0), 1.0),
                        '_0_wcmrss__08__02_P':hp.uniform('_0_wcmrss__08__02_P', 0.1, 1.0),
                        '_0_wcmrss__08__03_S':hp.choice('_0_wcmrss__08__03_S', ['1', ]),
                    },
                ]),
                hp.choice('meta_base_choice', [
                    {
                        '_1_wcbbn_W':'weka.classifiers.bayes.BayesNet',
                        '_1_wcbbn_W-':'REMOVED',
                        '_1_wcbbn_W_00__02_D':hp.choice('_1_wcbbn_W_00__02_D', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcbbn_W_00__03_Q':hp.choice('_1_wcbbn_W_00__03_Q', ['weka.classifiers.bayes.net.search.local.TAN', 'weka.classifiers.bayes.net.search.local.HillClimber', 'weka.classifiers.bayes.net.search.local.LAGDHillClimber', 'weka.classifiers.bayes.net.search.local.SimulatedAnnealing', 'weka.classifiers.bayes.net.search.local.TabuSearch', 'weka.classifiers.bayes.net.search.local.K2', ]),
                    },
                    {
                        '_1_wcbnb_W':'weka.classifiers.bayes.NaiveBayes',
                        '_1_wcbnb_W-':'REMOVED',
                        '_1_wcbnb_W_01__02_K':hp.choice('_1_wcbnb_W_01__02_K', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcbnb_W_01__03_D':hp.choice('_1_wcbnb_W_01__03_D', ['REMOVE_PREV', 'REMOVED', ]),
                    },
                    {
                        '_1_wcbnbm_W':'weka.classifiers.bayes.NaiveBayesMultinomial',
                        '_1_wcbnbm_W-':'REMOVED',
                    },
                    {
                        '_1_wcfl_W':'weka.classifiers.functions.Logistic',
                        '_1_wcfl_W-':'REMOVED',
                        '_1_wcfl_W_03__02_R':hp.loguniform('_1_wcfl_W_03__02_R', math.log(1.0E-12), math.log(10.0)),
                    },
                    {
                        '_1_wcfmp_W':'weka.classifiers.functions.MultilayerPerceptron',
                        '_1_wcfmp_W-':'REMOVED',
                        '_1_wcfmp_W_04__02_L':hp.uniform('_1_wcfmp_W_04__02_L', 0.1, 1.0),
                        '_1_wcfmp_W_04__03_M':hp.uniform('_1_wcfmp_W_04__03_M', 0.1, 1.0),
                        '_1_wcfmp_W_04__04_B':hp.choice('_1_wcfmp_W_04__04_B', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcfmp_W_04__05_H':hp.choice('_1_wcfmp_W_04__05_H', ['t', 'i', 'o', 'a', ]),
                        '_1_wcfmp_W_04__06_C':hp.choice('_1_wcfmp_W_04__06_C', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcfmp_W_04__07_R':hp.choice('_1_wcfmp_W_04__07_R', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcfmp_W_04__08_D':hp.choice('_1_wcfmp_W_04__08_D', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcfmp_W_04__09_S':hp.choice('_1_wcfmp_W_04__09_S', ['1', ]),
                    },
                    {
                        '_1_wcfsgd_W':'weka.classifiers.functions.SGD',
                        '_1_wcfsgd_W-':'REMOVED',
                        '_1_wcfsgd_W_05__02_F':hp.choice('_1_wcfsgd_W_05__02_F', ['2', '1', '0', ]),
                        '_1_wcfsgd_W_05__03_L':hp.loguniform('_1_wcfsgd_W_05__03_L', math.log(1.0E-5), math.log(0.1)),
                        '_1_wcfsgd_W_05__04_R':hp.loguniform('_1_wcfsgd_W_05__04_R', math.log(1.0E-12), math.log(10.0)),
                        '_1_wcfsgd_W_05__05_N':hp.choice('_1_wcfsgd_W_05__05_N', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcfsgd_W_05__06_M':hp.choice('_1_wcfsgd_W_05__06_M', ['REMOVE_PREV', 'REMOVED', ]),
                    },
                    {
                        '_1_wcfsmo_W':'weka.classifiers.functions.SMO',
                        '_1_wcfsmo_W-':'REMOVED',
                        '_1_wcfsmo_W_06__02_0_C':hp.uniform('_1_wcfsmo_W_06__02_0_C', 0.5, 1.5),
                        '_1_wcfsmo_W_06__03_1_N':hp.choice('_1_wcfsmo_W_06__03_1_N', ['2', '1', '0', ]),
                        '_1_wcfsmo_W_06__04_2_M':hp.choice('_1_wcfsmo_W_06__04_2_M', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcfsmo_W_06__05_5_QUOTE_END':hp.choice('_1_wcfsmo_W_06__05_5_QUOTE_END', ['REMOVED', ]),
                        '_1_wcfsmo_W_06__06_auto_param_106':hp.choice('_1_wcfsmo_W_06__06_auto_param_106', [
                            {
                                '_1_wcfsmo_W_06__06__00__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.NormalizedPolyKernel',
                                '_1_wcfsmo_W_06__06__00__01_4_npoly_E':hp.uniform('_1_wcfsmo_W_06__06__00__01_4_npoly_E', 0.2, 5.0),
                                '_1_wcfsmo_W_06__06__00__02_4_npoly_L':hp.choice('_1_wcfsmo_W_06__06__00__02_4_npoly_L', ['REMOVE_PREV', 'REMOVED', ]),
                            },
                            {
                                '_1_wcfsmo_W_06__06__01__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.PolyKernel',
                                '_1_wcfsmo_W_06__06__01__01_4_poly_E':hp.uniform('_1_wcfsmo_W_06__06__01__01_4_poly_E', 0.2, 5.0),
                                '_1_wcfsmo_W_06__06__01__02_4_poly_L':hp.choice('_1_wcfsmo_W_06__06__01__02_4_poly_L', ['REMOVE_PREV', 'REMOVED', ]),
                            },
                            {
                                '_1_wcfsmo_W_06__06__02__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.Puk',
                                '_1_wcfsmo_W_06__06__02__01_4_puk_S':hp.uniform('_1_wcfsmo_W_06__06__02__01_4_puk_S', 0.1, 10.0),
                                '_1_wcfsmo_W_06__06__02__02_4_puk_O':hp.uniform('_1_wcfsmo_W_06__06__02__02_4_puk_O', 0.1, 1.0),
                            },
                            {
                                '_1_wcfsmo_W_06__06__03__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.RBFKernel',
                                '_1_wcfsmo_W_06__06__03__01_4_rbf_G':hp.loguniform('_1_wcfsmo_W_06__06__03__01_4_rbf_G', math.log(1.0E-4), math.log(1.0)),
                            },
                        ]),
                    },
                    {
                        '_1_wcfsl_W':'weka.classifiers.functions.SimpleLogistic',
                        '_1_wcfsl_W-':'REMOVED',
                        '_1_wcfsl_W_07__02_S':hp.choice('_1_wcfsl_W_07__02_S', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcfsl_W_07__03_A':hp.choice('_1_wcfsl_W_07__03_A', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcfsl_W_07__04_auto_param_112':hp.choice('_1_wcfsl_W_07__04_auto_param_112', [
                            {
                                '_1_wcfsl_W_07__04__00__00_W_HIDDEN':'0',
                                '_1_wcfsl_W_07__04__00__01_1_W':hp.choice('_1_wcfsl_W_07__04__00__01_1_W', ['0', ]),
                            },
                            {
                                '_1_wcfsl_W_07__04__01__00_W_HIDDEN':'1',
                                '_1_wcfsl_W_07__04__01__01_2_W':hp.uniform('_1_wcfsl_W_07__04__01__01_2_W', 0.0, 1.0),
                            },
                        ]),
                    },
                    {
                        '_1_wcfvp_W':'weka.classifiers.functions.VotedPerceptron',
                        '_1_wcfvp_W-':'REMOVED',
                        '_1_wcfvp_W_08__02_INT_I':hp.quniform('_1_wcfvp_W_08__02_INT_I', 1.0, 10.0,1.0),
                        '_1_wcfvp_W_08__03_INT_M':hp.qloguniform('_1_wcfvp_W_08__03_INT_M', math.log(5000.0), math.log(50000.0), 1.0),
                        '_1_wcfvp_W_08__04_E':hp.uniform('_1_wcfvp_W_08__04_E', 0.2, 5.0),
                    },
                    {
                        '_1_wclib_W':'weka.classifiers.lazy.IBk',
                        '_1_wclib_W-':'REMOVED',
                        '_1_wclib_W_09__02_E':hp.choice('_1_wclib_W_09__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wclib_W_09__03_INT_K':hp.qloguniform('_1_wclib_W_09__03_INT_K', math.log(1.0), math.log(64.0), 1.0),
                        '_1_wclib_W_09__04_X':hp.choice('_1_wclib_W_09__04_X', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wclib_W_09__05_F':hp.choice('_1_wclib_W_09__05_F', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wclib_W_09__06_I':hp.choice('_1_wclib_W_09__06_I', ['REMOVE_PREV', 'REMOVED', ]),
                    },
                    {
                        '_1_wclks_W':'weka.classifiers.lazy.KStar',
                        '_1_wclks_W-':'REMOVED',
                        '_1_wclks_W_10__02_INT_B':hp.quniform('_1_wclks_W_10__02_INT_B', 1.0, 100.0,1.0),
                        '_1_wclks_W_10__03_E':hp.choice('_1_wclks_W_10__03_E', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wclks_W_10__04_M':hp.choice('_1_wclks_W_10__04_M', ['n', 'd', 'm', 'a', ]),
                    },
                    {
                        '_1_wcrdt_W':'weka.classifiers.rules.DecisionTable',
                        '_1_wcrdt_W-':'REMOVED',
                        '_1_wcrdt_W_11__02_E':hp.choice('_1_wcrdt_W_11__02_E', ['auc', 'rmse', 'mae', 'acc', ]),
                        '_1_wcrdt_W_11__03_I':hp.choice('_1_wcrdt_W_11__03_I', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcrdt_W_11__04_S':hp.choice('_1_wcrdt_W_11__04_S', ['weka.attributeSelection.Ranker', 'weka.attributeSelection.GreedyStepwise', 'weka.attributeSelection.BestFirst', ]),
                        '_1_wcrdt_W_11__05_X':hp.choice('_1_wcrdt_W_11__05_X', ['4', '2', '3', '1', ]),
                    },
                    {
                        '_1_wcrjr_W':'weka.classifiers.rules.JRip',
                        '_1_wcrjr_W-':'REMOVED',
                        '_1_wcrjr_W_12__02_N':hp.uniform('_1_wcrjr_W_12__02_N', 1.0, 5.0),
                        '_1_wcrjr_W_12__03_E':hp.choice('_1_wcrjr_W_12__03_E', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcrjr_W_12__04_P':hp.choice('_1_wcrjr_W_12__04_P', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcrjr_W_12__05_INT_O':hp.quniform('_1_wcrjr_W_12__05_INT_O', 1.0, 5.0,1.0),
                    },
                    {
                        '_1_wcror_W':'weka.classifiers.rules.OneR',
                        '_1_wcror_W-':'REMOVED',
                        '_1_wcror_W_13__02_INT_B':hp.qloguniform('_1_wcror_W_13__02_INT_B', math.log(1.0), math.log(32.0), 1.0),
                    },
                    {
                        '_1_wcrpart_W':'weka.classifiers.rules.PART',
                        '_1_wcrpart_W-':'REMOVED',
                        '_1_wcrpart_W_14__02_INT_N':hp.quniform('_1_wcrpart_W_14__02_INT_N', 2.0, 5.0,1.0),
                        '_1_wcrpart_W_14__03_INT_M':hp.qloguniform('_1_wcrpart_W_14__03_INT_M', math.log(1.0), math.log(64.0), 1.0),
                        '_1_wcrpart_W_14__04_R':hp.choice('_1_wcrpart_W_14__04_R', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wcrpart_W_14__05_B':hp.choice('_1_wcrpart_W_14__05_B', ['REMOVE_PREV', 'REMOVED', ]),
                    },
                    {
                        '_1_wcrzr_W':'weka.classifiers.rules.ZeroR',
                        '_1_wcrzr_W-':'REMOVED',
                    },
                    {
                        '_1_wctds_W':'weka.classifiers.trees.DecisionStump',
                        '_1_wctds_W-':'REMOVED',
                    },
                    {
                        '_1_wctj_W':'weka.classifiers.trees.J48',
                        '_1_wctj_W-':'REMOVED',
                        '_1_wctj_W_17__02_O':hp.choice('_1_wctj_W_17__02_O', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctj_W_17__03_U':hp.choice('_1_wctj_W_17__03_U', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctj_W_17__04_B':hp.choice('_1_wctj_W_17__04_B', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctj_W_17__05_J':hp.choice('_1_wctj_W_17__05_J', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctj_W_17__06_A':hp.choice('_1_wctj_W_17__06_A', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctj_W_17__07_S':hp.choice('_1_wctj_W_17__07_S', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctj_W_17__08_INT_M':hp.qloguniform('_1_wctj_W_17__08_INT_M', math.log(1.0), math.log(64.0), 1.0),
                        '_1_wctj_W_17__09_C':hp.uniform('_1_wctj_W_17__09_C', 0.0, 1.0),
                    },
                    {
                        '_1_wctlmt_W':'weka.classifiers.trees.LMT',
                        '_1_wctlmt_W-':'REMOVED',
                        '_1_wctlmt_W_18__02_B':hp.choice('_1_wctlmt_W_18__02_B', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctlmt_W_18__03_R':hp.choice('_1_wctlmt_W_18__03_R', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctlmt_W_18__04_C':hp.choice('_1_wctlmt_W_18__04_C', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctlmt_W_18__05_P':hp.choice('_1_wctlmt_W_18__05_P', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctlmt_W_18__06_INT_M':hp.qloguniform('_1_wctlmt_W_18__06_INT_M', math.log(1.0), math.log(64.0), 1.0),
                        '_1_wctlmt_W_18__07_A':hp.choice('_1_wctlmt_W_18__07_A', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctlmt_W_18__08_auto_param_126':hp.choice('_1_wctlmt_W_18__08_auto_param_126', [
                            {
                                '_1_wctlmt_W_18__08__00__00_W_HIDDEN':'0',
                                '_1_wctlmt_W_18__08__00__01_1_W':hp.choice('_1_wctlmt_W_18__08__00__01_1_W', ['0', ]),
                            },
                            {
                                '_1_wctlmt_W_18__08__01__00_W_HIDDEN':'1',
                                '_1_wctlmt_W_18__08__01__01_2_W':hp.uniform('_1_wctlmt_W_18__08__01__01_2_W', 0.0, 1.0),
                            },
                        ]),
                    },
                    {
                        '_1_wctrept_W':'weka.classifiers.trees.REPTree',
                        '_1_wctrept_W-':'REMOVED',
                        '_1_wctrept_W_19__02_INT_M':hp.qloguniform('_1_wctrept_W_19__02_INT_M', math.log(1.0), math.log(64.0), 1.0),
                        '_1_wctrept_W_19__03_V':hp.loguniform('_1_wctrept_W_19__03_V', math.log(1.0E-5), math.log(0.1)),
                        '_1_wctrept_W_19__04_P':hp.choice('_1_wctrept_W_19__04_P', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctrept_W_19__05_auto_param_130':hp.choice('_1_wctrept_W_19__05_auto_param_130', [
                            {
                                '_1_wctrept_W_19__05__00__00_depth_HIDDEN':'0',
                                '_1_wctrept_W_19__05__00__01_1_INT_L':hp.choice('_1_wctrept_W_19__05__00__01_1_INT_L', ['-1', ]),
                            },
                            {
                                '_1_wctrept_W_19__05__01__00_depth_HIDDEN':'1',
                                '_1_wctrept_W_19__05__01__01_2_INT_L':hp.quniform('_1_wctrept_W_19__05__01__01_2_INT_L', 2.0, 20.0,1.0),
                            },
                        ]),
                    },
                    {
                        '_1_wctrf_W':'weka.classifiers.trees.RandomForest',
                        '_1_wctrf_W-':'REMOVED',
                        '_1_wctrf_W_20__02_INT_I':hp.qloguniform('_1_wctrf_W_20__02_INT_I', math.log(2.0), math.log(256.0), 1.0),
                        '_1_wctrf_W_20__03_auto_param_134':hp.choice('_1_wctrf_W_20__03_auto_param_134', [
                            {
                                '_1_wctrf_W_20__03__00__00_features_HIDDEN':'0',
                                '_1_wctrf_W_20__03__00__01_1_INT_K':hp.choice('_1_wctrf_W_20__03__00__01_1_INT_K', ['1', ]),
                            },
                            {
                                '_1_wctrf_W_20__03__01__00_features_HIDDEN':'1',
                                '_1_wctrf_W_20__03__01__01_2_INT_K':hp.qloguniform('_1_wctrf_W_20__03__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                            },
                        ]),
                        '_1_wctrf_W_20__04_auto_param_137':hp.choice('_1_wctrf_W_20__04_auto_param_137', [
                            {
                                '_1_wctrf_W_20__04__00__00_depth_HIDDEN':'0',
                                '_1_wctrf_W_20__04__00__01_1_INT_depth':hp.choice('_1_wctrf_W_20__04__00__01_1_INT_depth', ['1', ]),
                            },
                            {
                                '_1_wctrf_W_20__04__01__00_depth_HIDDEN':'1',
                                '_1_wctrf_W_20__04__01__01_2_INT_depth':hp.quniform('_1_wctrf_W_20__04__01__01_2_INT_depth', 2.0, 20.0,1.0),
                            },
                        ]),
                    },
                    {
                        '_1_wctrt_W':'weka.classifiers.trees.RandomTree',
                        '_1_wctrt_W-':'REMOVED',
                        '_1_wctrt_W_21__02_INT_M':hp.qloguniform('_1_wctrt_W_21__02_INT_M', math.log(1.0), math.log(64.0), 1.0),
                        '_1_wctrt_W_21__03_U':hp.choice('_1_wctrt_W_21__03_U', ['REMOVE_PREV', 'REMOVED', ]),
                        '_1_wctrt_W_21__04_auto_param_141':hp.choice('_1_wctrt_W_21__04_auto_param_141', [
                            {
                                '_1_wctrt_W_21__04__00__00_features_HIDDEN':'0',
                                '_1_wctrt_W_21__04__00__01_1_INT_K':hp.choice('_1_wctrt_W_21__04__00__01_1_INT_K', ['0', ]),
                            },
                            {
                                '_1_wctrt_W_21__04__01__00_features_HIDDEN':'1',
                                '_1_wctrt_W_21__04__01__01_2_INT_K':hp.qloguniform('_1_wctrt_W_21__04__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                            },
                        ]),
                        '_1_wctrt_W_21__05_auto_param_144':hp.choice('_1_wctrt_W_21__05_auto_param_144', [
                            {
                                '_1_wctrt_W_21__05__00__00_depth_HIDDEN':'0',
                                '_1_wctrt_W_21__05__00__01_1_INT_depth':hp.choice('_1_wctrt_W_21__05__00__01_1_INT_depth', ['0', ]),
                            },
                            {
                                '_1_wctrt_W_21__05__01__00_depth_HIDDEN':'1',
                                '_1_wctrt_W_21__05__01__01_2_INT_depth':hp.quniform('_1_wctrt_W_21__05__01__01_2_INT_depth', 2.0, 20.0,1.0),
                            },
                        ]),
                        '_1_wctrt_W_21__06_auto_param_147':hp.choice('_1_wctrt_W_21__06_auto_param_147', [
                            {
                                '_1_wctrt_W_21__06__00__00_back_HIDDEN':'0',
                                '_1_wctrt_W_21__06__00__01_1_INT_N':hp.choice('_1_wctrt_W_21__06__00__01_1_INT_N', ['0', ]),
                            },
                            {
                                '_1_wctrt_W_21__06__01__00_back_HIDDEN':'1',
                                '_1_wctrt_W_21__06__01__01_2_INT_N':hp.quniform('_1_wctrt_W_21__06__01__01_2_INT_N', 2.0, 5.0,1.0),
                            },
                        ]),
                    },
                ]),
            ),
            (
                hp.choice('ensemble_choice', [
                    {
                        'targetclass':'weka.classifiers.meta.Stacking',
                        '_0_wcms__00__01_X':hp.choice('_0_wcms__00__01_X', ['10', ]),
                        '_0_wcms__00__02_S':hp.choice('_0_wcms__00__02_S', ['1', ]),
                    },
                    {
                        'targetclass':'weka.classifiers.meta.Vote',
                        '_0_wcmv__01__01_R':hp.choice('_0_wcmv__01__01_R', ['MAX', 'PROD', 'MAJ', 'MIN', 'AVG', ]),
                        '_0_wcmv__01__02_S':hp.choice('_0_wcmv__01__02_S', ['1', ]),
                    },
                ]),
                (
                    hp.choice('ensemble_base_choice', [
                        {
                            '_1_00_wcbbn_0_QUOTE_START_B':'weka.classifiers.bayes.BayesNet',
                            '_1_00_wcbbn_1__00__01_D':hp.choice('_1_00_wcbbn_1__00__01_D', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcbbn_1__00__02_Q':hp.choice('_1_00_wcbbn_1__00__02_Q', ['weka.classifiers.bayes.net.search.local.TAN', 'weka.classifiers.bayes.net.search.local.HillClimber', 'weka.classifiers.bayes.net.search.local.LAGDHillClimber', 'weka.classifiers.bayes.net.search.local.SimulatedAnnealing', 'weka.classifiers.bayes.net.search.local.TabuSearch', 'weka.classifiers.bayes.net.search.local.K2', ]),
                            '_1_00_wcbbn_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcbnb_0_QUOTE_START_B':'weka.classifiers.bayes.NaiveBayes',
                            '_1_00_wcbnb_1__01__01_K':hp.choice('_1_00_wcbnb_1__01__01_K', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcbnb_1__01__02_D':hp.choice('_1_00_wcbnb_1__01__02_D', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcbnb_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcbnbm_0_QUOTE_START_B':'weka.classifiers.bayes.NaiveBayesMultinomial',
                            '_1_00_wcbnbm_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcfl_0_QUOTE_START_B':'weka.classifiers.functions.Logistic',
                            '_1_00_wcfl_1__03__01_R':hp.loguniform('_1_00_wcfl_1__03__01_R', math.log(1.0E-12), math.log(10.0)),
                            '_1_00_wcfl_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcfmp_0_QUOTE_START_B':'weka.classifiers.functions.MultilayerPerceptron',
                            '_1_00_wcfmp_1__04__01_L':hp.uniform('_1_00_wcfmp_1__04__01_L', 0.1, 1.0),
                            '_1_00_wcfmp_1__04__02_M':hp.uniform('_1_00_wcfmp_1__04__02_M', 0.1, 1.0),
                            '_1_00_wcfmp_1__04__03_B':hp.choice('_1_00_wcfmp_1__04__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcfmp_1__04__04_H':hp.choice('_1_00_wcfmp_1__04__04_H', ['t', 'i', 'o', 'a', ]),
                            '_1_00_wcfmp_1__04__05_C':hp.choice('_1_00_wcfmp_1__04__05_C', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcfmp_1__04__06_R':hp.choice('_1_00_wcfmp_1__04__06_R', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcfmp_1__04__07_D':hp.choice('_1_00_wcfmp_1__04__07_D', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcfmp_1__04__08_S':hp.choice('_1_00_wcfmp_1__04__08_S', ['1', ]),
                            '_1_00_wcfmp_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcfsgd_0_QUOTE_START_B':'weka.classifiers.functions.SGD',
                            '_1_00_wcfsgd_1__05__01_F':hp.choice('_1_00_wcfsgd_1__05__01_F', ['2', '1', '0', ]),
                            '_1_00_wcfsgd_1__05__02_L':hp.loguniform('_1_00_wcfsgd_1__05__02_L', math.log(1.0E-5), math.log(0.1)),
                            '_1_00_wcfsgd_1__05__03_R':hp.loguniform('_1_00_wcfsgd_1__05__03_R', math.log(1.0E-12), math.log(10.0)),
                            '_1_00_wcfsgd_1__05__04_N':hp.choice('_1_00_wcfsgd_1__05__04_N', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcfsgd_1__05__05_M':hp.choice('_1_00_wcfsgd_1__05__05_M', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcfsgd_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcfsmo_0_QUOTE_START_B':'weka.classifiers.functions.SMO',
                            '_1_00_wcfsmo_1__06__01_0_C':hp.uniform('_1_00_wcfsmo_1__06__01_0_C', 0.5, 1.5),
                            '_1_00_wcfsmo_1__06__02_1_N':hp.choice('_1_00_wcfsmo_1__06__02_1_N', ['2', '1', '0', ]),
                            '_1_00_wcfsmo_1__06__03_2_M':hp.choice('_1_00_wcfsmo_1__06__03_2_M', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcfsmo_1__06__04_5_QUOTE_END':hp.choice('_1_00_wcfsmo_1__06__04_5_QUOTE_END', ['REMOVED', ]),
                            '_1_00_wcfsmo_1__06__05_auto_param_161':hp.choice('_1_00_wcfsmo_1__06__05_auto_param_161', [
                                {
                                    '_1_00_wcfsmo_1__06__05__00__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.NormalizedPolyKernel',
                                    '_1_00_wcfsmo_1__06__05__00__01_4_npoly_E':hp.uniform('_1_00_wcfsmo_1__06__05__00__01_4_npoly_E', 0.2, 5.0),
                                    '_1_00_wcfsmo_1__06__05__00__02_4_npoly_L':hp.choice('_1_00_wcfsmo_1__06__05__00__02_4_npoly_L', ['REMOVE_PREV', 'REMOVED', ]),
                                },
                                {
                                    '_1_00_wcfsmo_1__06__05__01__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.PolyKernel',
                                    '_1_00_wcfsmo_1__06__05__01__01_4_poly_E':hp.uniform('_1_00_wcfsmo_1__06__05__01__01_4_poly_E', 0.2, 5.0),
                                    '_1_00_wcfsmo_1__06__05__01__02_4_poly_L':hp.choice('_1_00_wcfsmo_1__06__05__01__02_4_poly_L', ['REMOVE_PREV', 'REMOVED', ]),
                                },
                                {
                                    '_1_00_wcfsmo_1__06__05__02__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.Puk',
                                    '_1_00_wcfsmo_1__06__05__02__01_4_puk_S':hp.uniform('_1_00_wcfsmo_1__06__05__02__01_4_puk_S', 0.1, 10.0),
                                    '_1_00_wcfsmo_1__06__05__02__02_4_puk_O':hp.uniform('_1_00_wcfsmo_1__06__05__02__02_4_puk_O', 0.1, 1.0),
                                },
                                {
                                    '_1_00_wcfsmo_1__06__05__03__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.RBFKernel',
                                    '_1_00_wcfsmo_1__06__05__03__01_4_rbf_G':hp.loguniform('_1_00_wcfsmo_1__06__05__03__01_4_rbf_G', math.log(1.0E-4), math.log(1.0)),
                                },
                            ]),
                            '_1_00_wcfsmo_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcfsl_0_QUOTE_START_B':'weka.classifiers.functions.SimpleLogistic',
                            '_1_00_wcfsl_1__07__01_S':hp.choice('_1_00_wcfsl_1__07__01_S', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcfsl_1__07__02_A':hp.choice('_1_00_wcfsl_1__07__02_A', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcfsl_1__07__03_auto_param_167':hp.choice('_1_00_wcfsl_1__07__03_auto_param_167', [
                                {
                                    '_1_00_wcfsl_1__07__03__00__00_W_HIDDEN':'0',
                                    '_1_00_wcfsl_1__07__03__00__01_1_W':hp.choice('_1_00_wcfsl_1__07__03__00__01_1_W', ['0', ]),
                                },
                                {
                                    '_1_00_wcfsl_1__07__03__01__00_W_HIDDEN':'1',
                                    '_1_00_wcfsl_1__07__03__01__01_2_W':hp.uniform('_1_00_wcfsl_1__07__03__01__01_2_W', 0.0, 1.0),
                                },
                            ]),
                            '_1_00_wcfsl_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcfvp_0_QUOTE_START_B':'weka.classifiers.functions.VotedPerceptron',
                            '_1_00_wcfvp_1__08__01_INT_I':hp.quniform('_1_00_wcfvp_1__08__01_INT_I', 1.0, 10.0,1.0),
                            '_1_00_wcfvp_1__08__02_INT_M':hp.qloguniform('_1_00_wcfvp_1__08__02_INT_M', math.log(5000.0), math.log(50000.0), 1.0),
                            '_1_00_wcfvp_1__08__03_E':hp.uniform('_1_00_wcfvp_1__08__03_E', 0.2, 5.0),
                            '_1_00_wcfvp_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wclib_0_QUOTE_START_B':'weka.classifiers.lazy.IBk',
                            '_1_00_wclib_1__09__01_E':hp.choice('_1_00_wclib_1__09__01_E', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wclib_1__09__02_INT_K':hp.qloguniform('_1_00_wclib_1__09__02_INT_K', math.log(1.0), math.log(64.0), 1.0),
                            '_1_00_wclib_1__09__03_X':hp.choice('_1_00_wclib_1__09__03_X', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wclib_1__09__04_F':hp.choice('_1_00_wclib_1__09__04_F', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wclib_1__09__05_I':hp.choice('_1_00_wclib_1__09__05_I', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wclib_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wclks_0_QUOTE_START_B':'weka.classifiers.lazy.KStar',
                            '_1_00_wclks_1__10__01_INT_B':hp.quniform('_1_00_wclks_1__10__01_INT_B', 1.0, 100.0,1.0),
                            '_1_00_wclks_1__10__02_E':hp.choice('_1_00_wclks_1__10__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wclks_1__10__03_M':hp.choice('_1_00_wclks_1__10__03_M', ['n', 'd', 'm', 'a', ]),
                            '_1_00_wclks_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcrdt_0_QUOTE_START_B':'weka.classifiers.rules.DecisionTable',
                            '_1_00_wcrdt_1__11__01_E':hp.choice('_1_00_wcrdt_1__11__01_E', ['auc', 'rmse', 'mae', 'acc', ]),
                            '_1_00_wcrdt_1__11__02_I':hp.choice('_1_00_wcrdt_1__11__02_I', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcrdt_1__11__03_S':hp.choice('_1_00_wcrdt_1__11__03_S', ['weka.attributeSelection.Ranker', 'weka.attributeSelection.GreedyStepwise', 'weka.attributeSelection.BestFirst', ]),
                            '_1_00_wcrdt_1__11__04_X':hp.choice('_1_00_wcrdt_1__11__04_X', ['4', '2', '3', '1', ]),
                            '_1_00_wcrdt_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcrjr_0_QUOTE_START_B':'weka.classifiers.rules.JRip',
                            '_1_00_wcrjr_1__12__01_N':hp.uniform('_1_00_wcrjr_1__12__01_N', 1.0, 5.0),
                            '_1_00_wcrjr_1__12__02_E':hp.choice('_1_00_wcrjr_1__12__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcrjr_1__12__03_P':hp.choice('_1_00_wcrjr_1__12__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcrjr_1__12__04_INT_O':hp.quniform('_1_00_wcrjr_1__12__04_INT_O', 1.0, 5.0,1.0),
                            '_1_00_wcrjr_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcror_0_QUOTE_START_B':'weka.classifiers.rules.OneR',
                            '_1_00_wcror_1__13__01_INT_B':hp.qloguniform('_1_00_wcror_1__13__01_INT_B', math.log(1.0), math.log(32.0), 1.0),
                            '_1_00_wcror_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcrpart_0_QUOTE_START_B':'weka.classifiers.rules.PART',
                            '_1_00_wcrpart_1__14__01_INT_N':hp.quniform('_1_00_wcrpart_1__14__01_INT_N', 2.0, 5.0,1.0),
                            '_1_00_wcrpart_1__14__02_INT_M':hp.qloguniform('_1_00_wcrpart_1__14__02_INT_M', math.log(1.0), math.log(64.0), 1.0),
                            '_1_00_wcrpart_1__14__03_R':hp.choice('_1_00_wcrpart_1__14__03_R', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcrpart_1__14__04_B':hp.choice('_1_00_wcrpart_1__14__04_B', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wcrpart_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wcrzr_0_QUOTE_START_B':'weka.classifiers.rules.ZeroR',
                            '_1_00_wcrzr_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wctds_0_QUOTE_START_B':'weka.classifiers.trees.DecisionStump',
                            '_1_00_wctds_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wctj_0_QUOTE_START_B':'weka.classifiers.trees.J48',
                            '_1_00_wctj_1__17__01_O':hp.choice('_1_00_wctj_1__17__01_O', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctj_1__17__02_U':hp.choice('_1_00_wctj_1__17__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctj_1__17__03_B':hp.choice('_1_00_wctj_1__17__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctj_1__17__04_J':hp.choice('_1_00_wctj_1__17__04_J', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctj_1__17__05_A':hp.choice('_1_00_wctj_1__17__05_A', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctj_1__17__06_S':hp.choice('_1_00_wctj_1__17__06_S', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctj_1__17__07_INT_M':hp.qloguniform('_1_00_wctj_1__17__07_INT_M', math.log(1.0), math.log(64.0), 1.0),
                            '_1_00_wctj_1__17__08_C':hp.uniform('_1_00_wctj_1__17__08_C', 0.0, 1.0),
                            '_1_00_wctj_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wctlmt_0_QUOTE_START_B':'weka.classifiers.trees.LMT',
                            '_1_00_wctlmt_1__18__01_B':hp.choice('_1_00_wctlmt_1__18__01_B', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctlmt_1__18__02_R':hp.choice('_1_00_wctlmt_1__18__02_R', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctlmt_1__18__03_C':hp.choice('_1_00_wctlmt_1__18__03_C', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctlmt_1__18__04_P':hp.choice('_1_00_wctlmt_1__18__04_P', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctlmt_1__18__05_INT_M':hp.qloguniform('_1_00_wctlmt_1__18__05_INT_M', math.log(1.0), math.log(64.0), 1.0),
                            '_1_00_wctlmt_1__18__06_A':hp.choice('_1_00_wctlmt_1__18__06_A', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctlmt_1__18__07_auto_param_181':hp.choice('_1_00_wctlmt_1__18__07_auto_param_181', [
                                {
                                    '_1_00_wctlmt_1__18__07__00__00_W_HIDDEN':'0',
                                    '_1_00_wctlmt_1__18__07__00__01_1_W':hp.choice('_1_00_wctlmt_1__18__07__00__01_1_W', ['0', ]),
                                },
                                {
                                    '_1_00_wctlmt_1__18__07__01__00_W_HIDDEN':'1',
                                    '_1_00_wctlmt_1__18__07__01__01_2_W':hp.uniform('_1_00_wctlmt_1__18__07__01__01_2_W', 0.0, 1.0),
                                },
                            ]),
                            '_1_00_wctlmt_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wctrept_0_QUOTE_START_B':'weka.classifiers.trees.REPTree',
                            '_1_00_wctrept_1__19__01_INT_M':hp.qloguniform('_1_00_wctrept_1__19__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                            '_1_00_wctrept_1__19__02_V':hp.loguniform('_1_00_wctrept_1__19__02_V', math.log(1.0E-5), math.log(0.1)),
                            '_1_00_wctrept_1__19__03_P':hp.choice('_1_00_wctrept_1__19__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctrept_1__19__04_auto_param_185':hp.choice('_1_00_wctrept_1__19__04_auto_param_185', [
                                {
                                    '_1_00_wctrept_1__19__04__00__00_depth_HIDDEN':'0',
                                    '_1_00_wctrept_1__19__04__00__01_1_INT_L':hp.choice('_1_00_wctrept_1__19__04__00__01_1_INT_L', ['-1', ]),
                                },
                                {
                                    '_1_00_wctrept_1__19__04__01__00_depth_HIDDEN':'1',
                                    '_1_00_wctrept_1__19__04__01__01_2_INT_L':hp.quniform('_1_00_wctrept_1__19__04__01__01_2_INT_L', 2.0, 20.0,1.0),
                                },
                            ]),
                            '_1_00_wctrept_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wctrf_0_QUOTE_START_B':'weka.classifiers.trees.RandomForest',
                            '_1_00_wctrf_1__20__01_INT_I':hp.qloguniform('_1_00_wctrf_1__20__01_INT_I', math.log(2.0), math.log(256.0), 1.0),
                            '_1_00_wctrf_1__20__02_auto_param_189':hp.choice('_1_00_wctrf_1__20__02_auto_param_189', [
                                {
                                    '_1_00_wctrf_1__20__02__00__00_features_HIDDEN':'0',
                                    '_1_00_wctrf_1__20__02__00__01_1_INT_K':hp.choice('_1_00_wctrf_1__20__02__00__01_1_INT_K', ['1', ]),
                                },
                                {
                                    '_1_00_wctrf_1__20__02__01__00_features_HIDDEN':'1',
                                    '_1_00_wctrf_1__20__02__01__01_2_INT_K':hp.qloguniform('_1_00_wctrf_1__20__02__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                                },
                            ]),
                            '_1_00_wctrf_1__20__03_auto_param_192':hp.choice('_1_00_wctrf_1__20__03_auto_param_192', [
                                {
                                    '_1_00_wctrf_1__20__03__00__00_depth_HIDDEN':'0',
                                    '_1_00_wctrf_1__20__03__00__01_1_INT_depth':hp.choice('_1_00_wctrf_1__20__03__00__01_1_INT_depth', ['1', ]),
                                },
                                {
                                    '_1_00_wctrf_1__20__03__01__00_depth_HIDDEN':'1',
                                    '_1_00_wctrf_1__20__03__01__01_2_INT_depth':hp.quniform('_1_00_wctrf_1__20__03__01__01_2_INT_depth', 2.0, 20.0,1.0),
                                },
                            ]),
                            '_1_00_wctrf_2_QUOTE_END':'REMOVED',
                        },
                        {
                            '_1_00_wctrt_0_QUOTE_START_B':'weka.classifiers.trees.RandomTree',
                            '_1_00_wctrt_1__21__01_INT_M':hp.qloguniform('_1_00_wctrt_1__21__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                            '_1_00_wctrt_1__21__02_U':hp.choice('_1_00_wctrt_1__21__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                            '_1_00_wctrt_1__21__03_auto_param_196':hp.choice('_1_00_wctrt_1__21__03_auto_param_196', [
                                {
                                    '_1_00_wctrt_1__21__03__00__00_features_HIDDEN':'0',
                                    '_1_00_wctrt_1__21__03__00__01_1_INT_K':hp.choice('_1_00_wctrt_1__21__03__00__01_1_INT_K', ['0', ]),
                                },
                                {
                                    '_1_00_wctrt_1__21__03__01__00_features_HIDDEN':'1',
                                    '_1_00_wctrt_1__21__03__01__01_2_INT_K':hp.qloguniform('_1_00_wctrt_1__21__03__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                                },
                            ]),
                            '_1_00_wctrt_1__21__04_auto_param_199':hp.choice('_1_00_wctrt_1__21__04_auto_param_199', [
                                {
                                    '_1_00_wctrt_1__21__04__00__00_depth_HIDDEN':'0',
                                    '_1_00_wctrt_1__21__04__00__01_1_INT_depth':hp.choice('_1_00_wctrt_1__21__04__00__01_1_INT_depth', ['0', ]),
                                },
                                {
                                    '_1_00_wctrt_1__21__04__01__00_depth_HIDDEN':'1',
                                    '_1_00_wctrt_1__21__04__01__01_2_INT_depth':hp.quniform('_1_00_wctrt_1__21__04__01__01_2_INT_depth', 2.0, 20.0,1.0),
                                },
                            ]),
                            '_1_00_wctrt_1__21__05_auto_param_202':hp.choice('_1_00_wctrt_1__21__05_auto_param_202', [
                                {
                                    '_1_00_wctrt_1__21__05__00__00_back_HIDDEN':'0',
                                    '_1_00_wctrt_1__21__05__00__01_1_INT_N':hp.choice('_1_00_wctrt_1__21__05__00__01_1_INT_N', ['0', ]),
                                },
                                {
                                    '_1_00_wctrt_1__21__05__01__00_back_HIDDEN':'1',
                                    '_1_00_wctrt_1__21__05__01__01_2_INT_N':hp.quniform('_1_00_wctrt_1__21__05__01__01_2_INT_N', 2.0, 5.0,1.0),
                                },
                            ]),
                            '_1_00_wctrt_2_QUOTE_END':'REMOVED',
                        },
                    ]),
                    hp.choice('auto_param_205', [
                        (
                        ),
                        (
                            hp.choice('auto_param_208', [
                                {
                                    '_1_01_wcbbn_0_QUOTE_START_B':'weka.classifiers.bayes.BayesNet',
                                    '_1_01_wcbbn_1__00__01_D':hp.choice('_1_01_wcbbn_1__00__01_D', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcbbn_1__00__02_Q':hp.choice('_1_01_wcbbn_1__00__02_Q', ['weka.classifiers.bayes.net.search.local.TAN', 'weka.classifiers.bayes.net.search.local.HillClimber', 'weka.classifiers.bayes.net.search.local.LAGDHillClimber', 'weka.classifiers.bayes.net.search.local.SimulatedAnnealing', 'weka.classifiers.bayes.net.search.local.TabuSearch', 'weka.classifiers.bayes.net.search.local.K2', ]),
                                    '_1_01_wcbbn_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcbnb_0_QUOTE_START_B':'weka.classifiers.bayes.NaiveBayes',
                                    '_1_01_wcbnb_1__01__01_K':hp.choice('_1_01_wcbnb_1__01__01_K', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcbnb_1__01__02_D':hp.choice('_1_01_wcbnb_1__01__02_D', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcbnb_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcbnbm_0_QUOTE_START_B':'weka.classifiers.bayes.NaiveBayesMultinomial',
                                    '_1_01_wcbnbm_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcfl_0_QUOTE_START_B':'weka.classifiers.functions.Logistic',
                                    '_1_01_wcfl_1__03__01_R':hp.loguniform('_1_01_wcfl_1__03__01_R', math.log(1.0E-12), math.log(10.0)),
                                    '_1_01_wcfl_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcfmp_0_QUOTE_START_B':'weka.classifiers.functions.MultilayerPerceptron',
                                    '_1_01_wcfmp_1__04__01_L':hp.uniform('_1_01_wcfmp_1__04__01_L', 0.1, 1.0),
                                    '_1_01_wcfmp_1__04__02_M':hp.uniform('_1_01_wcfmp_1__04__02_M', 0.1, 1.0),
                                    '_1_01_wcfmp_1__04__03_B':hp.choice('_1_01_wcfmp_1__04__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcfmp_1__04__04_H':hp.choice('_1_01_wcfmp_1__04__04_H', ['t', 'i', 'o', 'a', ]),
                                    '_1_01_wcfmp_1__04__05_C':hp.choice('_1_01_wcfmp_1__04__05_C', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcfmp_1__04__06_R':hp.choice('_1_01_wcfmp_1__04__06_R', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcfmp_1__04__07_D':hp.choice('_1_01_wcfmp_1__04__07_D', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcfmp_1__04__08_S':hp.choice('_1_01_wcfmp_1__04__08_S', ['1', ]),
                                    '_1_01_wcfmp_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcfsgd_0_QUOTE_START_B':'weka.classifiers.functions.SGD',
                                    '_1_01_wcfsgd_1__05__01_F':hp.choice('_1_01_wcfsgd_1__05__01_F', ['2', '1', '0', ]),
                                    '_1_01_wcfsgd_1__05__02_L':hp.loguniform('_1_01_wcfsgd_1__05__02_L', math.log(1.0E-5), math.log(0.1)),
                                    '_1_01_wcfsgd_1__05__03_R':hp.loguniform('_1_01_wcfsgd_1__05__03_R', math.log(1.0E-12), math.log(10.0)),
                                    '_1_01_wcfsgd_1__05__04_N':hp.choice('_1_01_wcfsgd_1__05__04_N', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcfsgd_1__05__05_M':hp.choice('_1_01_wcfsgd_1__05__05_M', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcfsgd_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcfsmo_0_QUOTE_START_B':'weka.classifiers.functions.SMO',
                                    '_1_01_wcfsmo_1__06__01_0_C':hp.uniform('_1_01_wcfsmo_1__06__01_0_C', 0.5, 1.5),
                                    '_1_01_wcfsmo_1__06__02_1_N':hp.choice('_1_01_wcfsmo_1__06__02_1_N', ['2', '1', '0', ]),
                                    '_1_01_wcfsmo_1__06__03_2_M':hp.choice('_1_01_wcfsmo_1__06__03_2_M', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcfsmo_1__06__04_5_QUOTE_END':hp.choice('_1_01_wcfsmo_1__06__04_5_QUOTE_END', ['REMOVED', ]),
                                    '_1_01_wcfsmo_1__06__05_auto_param_216':hp.choice('_1_01_wcfsmo_1__06__05_auto_param_216', [
                                        {
                                            '_1_01_wcfsmo_1__06__05__00__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.NormalizedPolyKernel',
                                            '_1_01_wcfsmo_1__06__05__00__01_4_npoly_E':hp.uniform('_1_01_wcfsmo_1__06__05__00__01_4_npoly_E', 0.2, 5.0),
                                            '_1_01_wcfsmo_1__06__05__00__02_4_npoly_L':hp.choice('_1_01_wcfsmo_1__06__05__00__02_4_npoly_L', ['REMOVE_PREV', 'REMOVED', ]),
                                        },
                                        {
                                            '_1_01_wcfsmo_1__06__05__01__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.PolyKernel',
                                            '_1_01_wcfsmo_1__06__05__01__01_4_poly_E':hp.uniform('_1_01_wcfsmo_1__06__05__01__01_4_poly_E', 0.2, 5.0),
                                            '_1_01_wcfsmo_1__06__05__01__02_4_poly_L':hp.choice('_1_01_wcfsmo_1__06__05__01__02_4_poly_L', ['REMOVE_PREV', 'REMOVED', ]),
                                        },
                                        {
                                            '_1_01_wcfsmo_1__06__05__02__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.Puk',
                                            '_1_01_wcfsmo_1__06__05__02__01_4_puk_S':hp.uniform('_1_01_wcfsmo_1__06__05__02__01_4_puk_S', 0.1, 10.0),
                                            '_1_01_wcfsmo_1__06__05__02__02_4_puk_O':hp.uniform('_1_01_wcfsmo_1__06__05__02__02_4_puk_O', 0.1, 1.0),
                                        },
                                        {
                                            '_1_01_wcfsmo_1__06__05__03__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.RBFKernel',
                                            '_1_01_wcfsmo_1__06__05__03__01_4_rbf_G':hp.loguniform('_1_01_wcfsmo_1__06__05__03__01_4_rbf_G', math.log(1.0E-4), math.log(1.0)),
                                        },
                                    ]),
                                    '_1_01_wcfsmo_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcfsl_0_QUOTE_START_B':'weka.classifiers.functions.SimpleLogistic',
                                    '_1_01_wcfsl_1__07__01_S':hp.choice('_1_01_wcfsl_1__07__01_S', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcfsl_1__07__02_A':hp.choice('_1_01_wcfsl_1__07__02_A', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcfsl_1__07__03_auto_param_222':hp.choice('_1_01_wcfsl_1__07__03_auto_param_222', [
                                        {
                                            '_1_01_wcfsl_1__07__03__00__00_W_HIDDEN':'0',
                                            '_1_01_wcfsl_1__07__03__00__01_1_W':hp.choice('_1_01_wcfsl_1__07__03__00__01_1_W', ['0', ]),
                                        },
                                        {
                                            '_1_01_wcfsl_1__07__03__01__00_W_HIDDEN':'1',
                                            '_1_01_wcfsl_1__07__03__01__01_2_W':hp.uniform('_1_01_wcfsl_1__07__03__01__01_2_W', 0.0, 1.0),
                                        },
                                    ]),
                                    '_1_01_wcfsl_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcfvp_0_QUOTE_START_B':'weka.classifiers.functions.VotedPerceptron',
                                    '_1_01_wcfvp_1__08__01_INT_I':hp.quniform('_1_01_wcfvp_1__08__01_INT_I', 1.0, 10.0,1.0),
                                    '_1_01_wcfvp_1__08__02_INT_M':hp.qloguniform('_1_01_wcfvp_1__08__02_INT_M', math.log(5000.0), math.log(50000.0), 1.0),
                                    '_1_01_wcfvp_1__08__03_E':hp.uniform('_1_01_wcfvp_1__08__03_E', 0.2, 5.0),
                                    '_1_01_wcfvp_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wclib_0_QUOTE_START_B':'weka.classifiers.lazy.IBk',
                                    '_1_01_wclib_1__09__01_E':hp.choice('_1_01_wclib_1__09__01_E', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wclib_1__09__02_INT_K':hp.qloguniform('_1_01_wclib_1__09__02_INT_K', math.log(1.0), math.log(64.0), 1.0),
                                    '_1_01_wclib_1__09__03_X':hp.choice('_1_01_wclib_1__09__03_X', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wclib_1__09__04_F':hp.choice('_1_01_wclib_1__09__04_F', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wclib_1__09__05_I':hp.choice('_1_01_wclib_1__09__05_I', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wclib_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wclks_0_QUOTE_START_B':'weka.classifiers.lazy.KStar',
                                    '_1_01_wclks_1__10__01_INT_B':hp.quniform('_1_01_wclks_1__10__01_INT_B', 1.0, 100.0,1.0),
                                    '_1_01_wclks_1__10__02_E':hp.choice('_1_01_wclks_1__10__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wclks_1__10__03_M':hp.choice('_1_01_wclks_1__10__03_M', ['n', 'd', 'm', 'a', ]),
                                    '_1_01_wclks_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcrdt_0_QUOTE_START_B':'weka.classifiers.rules.DecisionTable',
                                    '_1_01_wcrdt_1__11__01_E':hp.choice('_1_01_wcrdt_1__11__01_E', ['auc', 'rmse', 'mae', 'acc', ]),
                                    '_1_01_wcrdt_1__11__02_I':hp.choice('_1_01_wcrdt_1__11__02_I', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcrdt_1__11__03_S':hp.choice('_1_01_wcrdt_1__11__03_S', ['weka.attributeSelection.Ranker', 'weka.attributeSelection.GreedyStepwise', 'weka.attributeSelection.BestFirst', ]),
                                    '_1_01_wcrdt_1__11__04_X':hp.choice('_1_01_wcrdt_1__11__04_X', ['4', '2', '3', '1', ]),
                                    '_1_01_wcrdt_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcrjr_0_QUOTE_START_B':'weka.classifiers.rules.JRip',
                                    '_1_01_wcrjr_1__12__01_N':hp.uniform('_1_01_wcrjr_1__12__01_N', 1.0, 5.0),
                                    '_1_01_wcrjr_1__12__02_E':hp.choice('_1_01_wcrjr_1__12__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcrjr_1__12__03_P':hp.choice('_1_01_wcrjr_1__12__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcrjr_1__12__04_INT_O':hp.quniform('_1_01_wcrjr_1__12__04_INT_O', 1.0, 5.0,1.0),
                                    '_1_01_wcrjr_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcror_0_QUOTE_START_B':'weka.classifiers.rules.OneR',
                                    '_1_01_wcror_1__13__01_INT_B':hp.qloguniform('_1_01_wcror_1__13__01_INT_B', math.log(1.0), math.log(32.0), 1.0),
                                    '_1_01_wcror_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcrpart_0_QUOTE_START_B':'weka.classifiers.rules.PART',
                                    '_1_01_wcrpart_1__14__01_INT_N':hp.quniform('_1_01_wcrpart_1__14__01_INT_N', 2.0, 5.0,1.0),
                                    '_1_01_wcrpart_1__14__02_INT_M':hp.qloguniform('_1_01_wcrpart_1__14__02_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                    '_1_01_wcrpart_1__14__03_R':hp.choice('_1_01_wcrpart_1__14__03_R', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcrpart_1__14__04_B':hp.choice('_1_01_wcrpart_1__14__04_B', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wcrpart_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wcrzr_0_QUOTE_START_B':'weka.classifiers.rules.ZeroR',
                                    '_1_01_wcrzr_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wctds_0_QUOTE_START_B':'weka.classifiers.trees.DecisionStump',
                                    '_1_01_wctds_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wctj_0_QUOTE_START_B':'weka.classifiers.trees.J48',
                                    '_1_01_wctj_1__17__01_O':hp.choice('_1_01_wctj_1__17__01_O', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctj_1__17__02_U':hp.choice('_1_01_wctj_1__17__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctj_1__17__03_B':hp.choice('_1_01_wctj_1__17__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctj_1__17__04_J':hp.choice('_1_01_wctj_1__17__04_J', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctj_1__17__05_A':hp.choice('_1_01_wctj_1__17__05_A', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctj_1__17__06_S':hp.choice('_1_01_wctj_1__17__06_S', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctj_1__17__07_INT_M':hp.qloguniform('_1_01_wctj_1__17__07_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                    '_1_01_wctj_1__17__08_C':hp.uniform('_1_01_wctj_1__17__08_C', 0.0, 1.0),
                                    '_1_01_wctj_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wctlmt_0_QUOTE_START_B':'weka.classifiers.trees.LMT',
                                    '_1_01_wctlmt_1__18__01_B':hp.choice('_1_01_wctlmt_1__18__01_B', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctlmt_1__18__02_R':hp.choice('_1_01_wctlmt_1__18__02_R', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctlmt_1__18__03_C':hp.choice('_1_01_wctlmt_1__18__03_C', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctlmt_1__18__04_P':hp.choice('_1_01_wctlmt_1__18__04_P', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctlmt_1__18__05_INT_M':hp.qloguniform('_1_01_wctlmt_1__18__05_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                    '_1_01_wctlmt_1__18__06_A':hp.choice('_1_01_wctlmt_1__18__06_A', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctlmt_1__18__07_auto_param_236':hp.choice('_1_01_wctlmt_1__18__07_auto_param_236', [
                                        {
                                            '_1_01_wctlmt_1__18__07__00__00_W_HIDDEN':'0',
                                            '_1_01_wctlmt_1__18__07__00__01_1_W':hp.choice('_1_01_wctlmt_1__18__07__00__01_1_W', ['0', ]),
                                        },
                                        {
                                            '_1_01_wctlmt_1__18__07__01__00_W_HIDDEN':'1',
                                            '_1_01_wctlmt_1__18__07__01__01_2_W':hp.uniform('_1_01_wctlmt_1__18__07__01__01_2_W', 0.0, 1.0),
                                        },
                                    ]),
                                    '_1_01_wctlmt_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wctrept_0_QUOTE_START_B':'weka.classifiers.trees.REPTree',
                                    '_1_01_wctrept_1__19__01_INT_M':hp.qloguniform('_1_01_wctrept_1__19__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                    '_1_01_wctrept_1__19__02_V':hp.loguniform('_1_01_wctrept_1__19__02_V', math.log(1.0E-5), math.log(0.1)),
                                    '_1_01_wctrept_1__19__03_P':hp.choice('_1_01_wctrept_1__19__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctrept_1__19__04_auto_param_240':hp.choice('_1_01_wctrept_1__19__04_auto_param_240', [
                                        {
                                            '_1_01_wctrept_1__19__04__00__00_depth_HIDDEN':'0',
                                            '_1_01_wctrept_1__19__04__00__01_1_INT_L':hp.choice('_1_01_wctrept_1__19__04__00__01_1_INT_L', ['-1', ]),
                                        },
                                        {
                                            '_1_01_wctrept_1__19__04__01__00_depth_HIDDEN':'1',
                                            '_1_01_wctrept_1__19__04__01__01_2_INT_L':hp.quniform('_1_01_wctrept_1__19__04__01__01_2_INT_L', 2.0, 20.0,1.0),
                                        },
                                    ]),
                                    '_1_01_wctrept_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wctrf_0_QUOTE_START_B':'weka.classifiers.trees.RandomForest',
                                    '_1_01_wctrf_1__20__01_INT_I':hp.qloguniform('_1_01_wctrf_1__20__01_INT_I', math.log(2.0), math.log(256.0), 1.0),
                                    '_1_01_wctrf_1__20__02_auto_param_244':hp.choice('_1_01_wctrf_1__20__02_auto_param_244', [
                                        {
                                            '_1_01_wctrf_1__20__02__00__00_features_HIDDEN':'0',
                                            '_1_01_wctrf_1__20__02__00__01_1_INT_K':hp.choice('_1_01_wctrf_1__20__02__00__01_1_INT_K', ['1', ]),
                                        },
                                        {
                                            '_1_01_wctrf_1__20__02__01__00_features_HIDDEN':'1',
                                            '_1_01_wctrf_1__20__02__01__01_2_INT_K':hp.qloguniform('_1_01_wctrf_1__20__02__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                                        },
                                    ]),
                                    '_1_01_wctrf_1__20__03_auto_param_247':hp.choice('_1_01_wctrf_1__20__03_auto_param_247', [
                                        {
                                            '_1_01_wctrf_1__20__03__00__00_depth_HIDDEN':'0',
                                            '_1_01_wctrf_1__20__03__00__01_1_INT_depth':hp.choice('_1_01_wctrf_1__20__03__00__01_1_INT_depth', ['1', ]),
                                        },
                                        {
                                            '_1_01_wctrf_1__20__03__01__00_depth_HIDDEN':'1',
                                            '_1_01_wctrf_1__20__03__01__01_2_INT_depth':hp.quniform('_1_01_wctrf_1__20__03__01__01_2_INT_depth', 2.0, 20.0,1.0),
                                        },
                                    ]),
                                    '_1_01_wctrf_2_QUOTE_END':'REMOVED',
                                },
                                {
                                    '_1_01_wctrt_0_QUOTE_START_B':'weka.classifiers.trees.RandomTree',
                                    '_1_01_wctrt_1__21__01_INT_M':hp.qloguniform('_1_01_wctrt_1__21__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                    '_1_01_wctrt_1__21__02_U':hp.choice('_1_01_wctrt_1__21__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                                    '_1_01_wctrt_1__21__03_auto_param_251':hp.choice('_1_01_wctrt_1__21__03_auto_param_251', [
                                        {
                                            '_1_01_wctrt_1__21__03__00__00_features_HIDDEN':'0',
                                            '_1_01_wctrt_1__21__03__00__01_1_INT_K':hp.choice('_1_01_wctrt_1__21__03__00__01_1_INT_K', ['0', ]),
                                        },
                                        {
                                            '_1_01_wctrt_1__21__03__01__00_features_HIDDEN':'1',
                                            '_1_01_wctrt_1__21__03__01__01_2_INT_K':hp.qloguniform('_1_01_wctrt_1__21__03__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                                        },
                                    ]),
                                    '_1_01_wctrt_1__21__04_auto_param_254':hp.choice('_1_01_wctrt_1__21__04_auto_param_254', [
                                        {
                                            '_1_01_wctrt_1__21__04__00__00_depth_HIDDEN':'0',
                                            '_1_01_wctrt_1__21__04__00__01_1_INT_depth':hp.choice('_1_01_wctrt_1__21__04__00__01_1_INT_depth', ['0', ]),
                                        },
                                        {
                                            '_1_01_wctrt_1__21__04__01__00_depth_HIDDEN':'1',
                                            '_1_01_wctrt_1__21__04__01__01_2_INT_depth':hp.quniform('_1_01_wctrt_1__21__04__01__01_2_INT_depth', 2.0, 20.0,1.0),
                                        },
                                    ]),
                                    '_1_01_wctrt_1__21__05_auto_param_257':hp.choice('_1_01_wctrt_1__21__05_auto_param_257', [
                                        {
                                            '_1_01_wctrt_1__21__05__00__00_back_HIDDEN':'0',
                                            '_1_01_wctrt_1__21__05__00__01_1_INT_N':hp.choice('_1_01_wctrt_1__21__05__00__01_1_INT_N', ['0', ]),
                                        },
                                        {
                                            '_1_01_wctrt_1__21__05__01__00_back_HIDDEN':'1',
                                            '_1_01_wctrt_1__21__05__01__01_2_INT_N':hp.quniform('_1_01_wctrt_1__21__05__01__01_2_INT_N', 2.0, 5.0,1.0),
                                        },
                                    ]),
                                    '_1_01_wctrt_2_QUOTE_END':'REMOVED',
                                },
                            ]),
                            hp.choice('auto_param_260', [
                                (
                                ),
                                (
                                    hp.choice('auto_param_263', [
                                        {
                                            '_1_02_wcbbn_0_QUOTE_START_B':'weka.classifiers.bayes.BayesNet',
                                            '_1_02_wcbbn_1__00__01_D':hp.choice('_1_02_wcbbn_1__00__01_D', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcbbn_1__00__02_Q':hp.choice('_1_02_wcbbn_1__00__02_Q', ['weka.classifiers.bayes.net.search.local.TAN', 'weka.classifiers.bayes.net.search.local.HillClimber', 'weka.classifiers.bayes.net.search.local.LAGDHillClimber', 'weka.classifiers.bayes.net.search.local.SimulatedAnnealing', 'weka.classifiers.bayes.net.search.local.TabuSearch', 'weka.classifiers.bayes.net.search.local.K2', ]),
                                            '_1_02_wcbbn_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcbnb_0_QUOTE_START_B':'weka.classifiers.bayes.NaiveBayes',
                                            '_1_02_wcbnb_1__01__01_K':hp.choice('_1_02_wcbnb_1__01__01_K', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcbnb_1__01__02_D':hp.choice('_1_02_wcbnb_1__01__02_D', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcbnb_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcbnbm_0_QUOTE_START_B':'weka.classifiers.bayes.NaiveBayesMultinomial',
                                            '_1_02_wcbnbm_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcfl_0_QUOTE_START_B':'weka.classifiers.functions.Logistic',
                                            '_1_02_wcfl_1__03__01_R':hp.loguniform('_1_02_wcfl_1__03__01_R', math.log(1.0E-12), math.log(10.0)),
                                            '_1_02_wcfl_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcfmp_0_QUOTE_START_B':'weka.classifiers.functions.MultilayerPerceptron',
                                            '_1_02_wcfmp_1__04__01_L':hp.uniform('_1_02_wcfmp_1__04__01_L', 0.1, 1.0),
                                            '_1_02_wcfmp_1__04__02_M':hp.uniform('_1_02_wcfmp_1__04__02_M', 0.1, 1.0),
                                            '_1_02_wcfmp_1__04__03_B':hp.choice('_1_02_wcfmp_1__04__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcfmp_1__04__04_H':hp.choice('_1_02_wcfmp_1__04__04_H', ['t', 'i', 'o', 'a', ]),
                                            '_1_02_wcfmp_1__04__05_C':hp.choice('_1_02_wcfmp_1__04__05_C', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcfmp_1__04__06_R':hp.choice('_1_02_wcfmp_1__04__06_R', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcfmp_1__04__07_D':hp.choice('_1_02_wcfmp_1__04__07_D', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcfmp_1__04__08_S':hp.choice('_1_02_wcfmp_1__04__08_S', ['1', ]),
                                            '_1_02_wcfmp_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcfsgd_0_QUOTE_START_B':'weka.classifiers.functions.SGD',
                                            '_1_02_wcfsgd_1__05__01_F':hp.choice('_1_02_wcfsgd_1__05__01_F', ['2', '1', '0', ]),
                                            '_1_02_wcfsgd_1__05__02_L':hp.loguniform('_1_02_wcfsgd_1__05__02_L', math.log(1.0E-5), math.log(0.1)),
                                            '_1_02_wcfsgd_1__05__03_R':hp.loguniform('_1_02_wcfsgd_1__05__03_R', math.log(1.0E-12), math.log(10.0)),
                                            '_1_02_wcfsgd_1__05__04_N':hp.choice('_1_02_wcfsgd_1__05__04_N', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcfsgd_1__05__05_M':hp.choice('_1_02_wcfsgd_1__05__05_M', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcfsgd_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcfsmo_0_QUOTE_START_B':'weka.classifiers.functions.SMO',
                                            '_1_02_wcfsmo_1__06__01_0_C':hp.uniform('_1_02_wcfsmo_1__06__01_0_C', 0.5, 1.5),
                                            '_1_02_wcfsmo_1__06__02_1_N':hp.choice('_1_02_wcfsmo_1__06__02_1_N', ['2', '1', '0', ]),
                                            '_1_02_wcfsmo_1__06__03_2_M':hp.choice('_1_02_wcfsmo_1__06__03_2_M', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcfsmo_1__06__04_5_QUOTE_END':hp.choice('_1_02_wcfsmo_1__06__04_5_QUOTE_END', ['REMOVED', ]),
                                            '_1_02_wcfsmo_1__06__05_auto_param_271':hp.choice('_1_02_wcfsmo_1__06__05_auto_param_271', [
                                                {
                                                    '_1_02_wcfsmo_1__06__05__00__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.NormalizedPolyKernel',
                                                    '_1_02_wcfsmo_1__06__05__00__01_4_npoly_E':hp.uniform('_1_02_wcfsmo_1__06__05__00__01_4_npoly_E', 0.2, 5.0),
                                                    '_1_02_wcfsmo_1__06__05__00__02_4_npoly_L':hp.choice('_1_02_wcfsmo_1__06__05__00__02_4_npoly_L', ['REMOVE_PREV', 'REMOVED', ]),
                                                },
                                                {
                                                    '_1_02_wcfsmo_1__06__05__01__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.PolyKernel',
                                                    '_1_02_wcfsmo_1__06__05__01__01_4_poly_E':hp.uniform('_1_02_wcfsmo_1__06__05__01__01_4_poly_E', 0.2, 5.0),
                                                    '_1_02_wcfsmo_1__06__05__01__02_4_poly_L':hp.choice('_1_02_wcfsmo_1__06__05__01__02_4_poly_L', ['REMOVE_PREV', 'REMOVED', ]),
                                                },
                                                {
                                                    '_1_02_wcfsmo_1__06__05__02__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.Puk',
                                                    '_1_02_wcfsmo_1__06__05__02__01_4_puk_S':hp.uniform('_1_02_wcfsmo_1__06__05__02__01_4_puk_S', 0.1, 10.0),
                                                    '_1_02_wcfsmo_1__06__05__02__02_4_puk_O':hp.uniform('_1_02_wcfsmo_1__06__05__02__02_4_puk_O', 0.1, 1.0),
                                                },
                                                {
                                                    '_1_02_wcfsmo_1__06__05__03__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.RBFKernel',
                                                    '_1_02_wcfsmo_1__06__05__03__01_4_rbf_G':hp.loguniform('_1_02_wcfsmo_1__06__05__03__01_4_rbf_G', math.log(1.0E-4), math.log(1.0)),
                                                },
                                            ]),
                                            '_1_02_wcfsmo_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcfsl_0_QUOTE_START_B':'weka.classifiers.functions.SimpleLogistic',
                                            '_1_02_wcfsl_1__07__01_S':hp.choice('_1_02_wcfsl_1__07__01_S', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcfsl_1__07__02_A':hp.choice('_1_02_wcfsl_1__07__02_A', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcfsl_1__07__03_auto_param_277':hp.choice('_1_02_wcfsl_1__07__03_auto_param_277', [
                                                {
                                                    '_1_02_wcfsl_1__07__03__00__00_W_HIDDEN':'0',
                                                    '_1_02_wcfsl_1__07__03__00__01_1_W':hp.choice('_1_02_wcfsl_1__07__03__00__01_1_W', ['0', ]),
                                                },
                                                {
                                                    '_1_02_wcfsl_1__07__03__01__00_W_HIDDEN':'1',
                                                    '_1_02_wcfsl_1__07__03__01__01_2_W':hp.uniform('_1_02_wcfsl_1__07__03__01__01_2_W', 0.0, 1.0),
                                                },
                                            ]),
                                            '_1_02_wcfsl_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcfvp_0_QUOTE_START_B':'weka.classifiers.functions.VotedPerceptron',
                                            '_1_02_wcfvp_1__08__01_INT_I':hp.quniform('_1_02_wcfvp_1__08__01_INT_I', 1.0, 10.0,1.0),
                                            '_1_02_wcfvp_1__08__02_INT_M':hp.qloguniform('_1_02_wcfvp_1__08__02_INT_M', math.log(5000.0), math.log(50000.0), 1.0),
                                            '_1_02_wcfvp_1__08__03_E':hp.uniform('_1_02_wcfvp_1__08__03_E', 0.2, 5.0),
                                            '_1_02_wcfvp_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wclib_0_QUOTE_START_B':'weka.classifiers.lazy.IBk',
                                            '_1_02_wclib_1__09__01_E':hp.choice('_1_02_wclib_1__09__01_E', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wclib_1__09__02_INT_K':hp.qloguniform('_1_02_wclib_1__09__02_INT_K', math.log(1.0), math.log(64.0), 1.0),
                                            '_1_02_wclib_1__09__03_X':hp.choice('_1_02_wclib_1__09__03_X', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wclib_1__09__04_F':hp.choice('_1_02_wclib_1__09__04_F', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wclib_1__09__05_I':hp.choice('_1_02_wclib_1__09__05_I', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wclib_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wclks_0_QUOTE_START_B':'weka.classifiers.lazy.KStar',
                                            '_1_02_wclks_1__10__01_INT_B':hp.quniform('_1_02_wclks_1__10__01_INT_B', 1.0, 100.0,1.0),
                                            '_1_02_wclks_1__10__02_E':hp.choice('_1_02_wclks_1__10__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wclks_1__10__03_M':hp.choice('_1_02_wclks_1__10__03_M', ['n', 'd', 'm', 'a', ]),
                                            '_1_02_wclks_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcrdt_0_QUOTE_START_B':'weka.classifiers.rules.DecisionTable',
                                            '_1_02_wcrdt_1__11__01_E':hp.choice('_1_02_wcrdt_1__11__01_E', ['auc', 'rmse', 'mae', 'acc', ]),
                                            '_1_02_wcrdt_1__11__02_I':hp.choice('_1_02_wcrdt_1__11__02_I', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcrdt_1__11__03_S':hp.choice('_1_02_wcrdt_1__11__03_S', ['weka.attributeSelection.Ranker', 'weka.attributeSelection.GreedyStepwise', 'weka.attributeSelection.BestFirst', ]),
                                            '_1_02_wcrdt_1__11__04_X':hp.choice('_1_02_wcrdt_1__11__04_X', ['4', '2', '3', '1', ]),
                                            '_1_02_wcrdt_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcrjr_0_QUOTE_START_B':'weka.classifiers.rules.JRip',
                                            '_1_02_wcrjr_1__12__01_N':hp.uniform('_1_02_wcrjr_1__12__01_N', 1.0, 5.0),
                                            '_1_02_wcrjr_1__12__02_E':hp.choice('_1_02_wcrjr_1__12__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcrjr_1__12__03_P':hp.choice('_1_02_wcrjr_1__12__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcrjr_1__12__04_INT_O':hp.quniform('_1_02_wcrjr_1__12__04_INT_O', 1.0, 5.0,1.0),
                                            '_1_02_wcrjr_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcror_0_QUOTE_START_B':'weka.classifiers.rules.OneR',
                                            '_1_02_wcror_1__13__01_INT_B':hp.qloguniform('_1_02_wcror_1__13__01_INT_B', math.log(1.0), math.log(32.0), 1.0),
                                            '_1_02_wcror_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcrpart_0_QUOTE_START_B':'weka.classifiers.rules.PART',
                                            '_1_02_wcrpart_1__14__01_INT_N':hp.quniform('_1_02_wcrpart_1__14__01_INT_N', 2.0, 5.0,1.0),
                                            '_1_02_wcrpart_1__14__02_INT_M':hp.qloguniform('_1_02_wcrpart_1__14__02_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                            '_1_02_wcrpart_1__14__03_R':hp.choice('_1_02_wcrpart_1__14__03_R', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcrpart_1__14__04_B':hp.choice('_1_02_wcrpart_1__14__04_B', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wcrpart_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wcrzr_0_QUOTE_START_B':'weka.classifiers.rules.ZeroR',
                                            '_1_02_wcrzr_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wctds_0_QUOTE_START_B':'weka.classifiers.trees.DecisionStump',
                                            '_1_02_wctds_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wctj_0_QUOTE_START_B':'weka.classifiers.trees.J48',
                                            '_1_02_wctj_1__17__01_O':hp.choice('_1_02_wctj_1__17__01_O', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctj_1__17__02_U':hp.choice('_1_02_wctj_1__17__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctj_1__17__03_B':hp.choice('_1_02_wctj_1__17__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctj_1__17__04_J':hp.choice('_1_02_wctj_1__17__04_J', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctj_1__17__05_A':hp.choice('_1_02_wctj_1__17__05_A', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctj_1__17__06_S':hp.choice('_1_02_wctj_1__17__06_S', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctj_1__17__07_INT_M':hp.qloguniform('_1_02_wctj_1__17__07_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                            '_1_02_wctj_1__17__08_C':hp.uniform('_1_02_wctj_1__17__08_C', 0.0, 1.0),
                                            '_1_02_wctj_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wctlmt_0_QUOTE_START_B':'weka.classifiers.trees.LMT',
                                            '_1_02_wctlmt_1__18__01_B':hp.choice('_1_02_wctlmt_1__18__01_B', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctlmt_1__18__02_R':hp.choice('_1_02_wctlmt_1__18__02_R', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctlmt_1__18__03_C':hp.choice('_1_02_wctlmt_1__18__03_C', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctlmt_1__18__04_P':hp.choice('_1_02_wctlmt_1__18__04_P', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctlmt_1__18__05_INT_M':hp.qloguniform('_1_02_wctlmt_1__18__05_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                            '_1_02_wctlmt_1__18__06_A':hp.choice('_1_02_wctlmt_1__18__06_A', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctlmt_1__18__07_auto_param_291':hp.choice('_1_02_wctlmt_1__18__07_auto_param_291', [
                                                {
                                                    '_1_02_wctlmt_1__18__07__00__00_W_HIDDEN':'0',
                                                    '_1_02_wctlmt_1__18__07__00__01_1_W':hp.choice('_1_02_wctlmt_1__18__07__00__01_1_W', ['0', ]),
                                                },
                                                {
                                                    '_1_02_wctlmt_1__18__07__01__00_W_HIDDEN':'1',
                                                    '_1_02_wctlmt_1__18__07__01__01_2_W':hp.uniform('_1_02_wctlmt_1__18__07__01__01_2_W', 0.0, 1.0),
                                                },
                                            ]),
                                            '_1_02_wctlmt_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wctrept_0_QUOTE_START_B':'weka.classifiers.trees.REPTree',
                                            '_1_02_wctrept_1__19__01_INT_M':hp.qloguniform('_1_02_wctrept_1__19__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                            '_1_02_wctrept_1__19__02_V':hp.loguniform('_1_02_wctrept_1__19__02_V', math.log(1.0E-5), math.log(0.1)),
                                            '_1_02_wctrept_1__19__03_P':hp.choice('_1_02_wctrept_1__19__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctrept_1__19__04_auto_param_295':hp.choice('_1_02_wctrept_1__19__04_auto_param_295', [
                                                {
                                                    '_1_02_wctrept_1__19__04__00__00_depth_HIDDEN':'0',
                                                    '_1_02_wctrept_1__19__04__00__01_1_INT_L':hp.choice('_1_02_wctrept_1__19__04__00__01_1_INT_L', ['-1', ]),
                                                },
                                                {
                                                    '_1_02_wctrept_1__19__04__01__00_depth_HIDDEN':'1',
                                                    '_1_02_wctrept_1__19__04__01__01_2_INT_L':hp.quniform('_1_02_wctrept_1__19__04__01__01_2_INT_L', 2.0, 20.0,1.0),
                                                },
                                            ]),
                                            '_1_02_wctrept_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wctrf_0_QUOTE_START_B':'weka.classifiers.trees.RandomForest',
                                            '_1_02_wctrf_1__20__01_INT_I':hp.qloguniform('_1_02_wctrf_1__20__01_INT_I', math.log(2.0), math.log(256.0), 1.0),
                                            '_1_02_wctrf_1__20__02_auto_param_299':hp.choice('_1_02_wctrf_1__20__02_auto_param_299', [
                                                {
                                                    '_1_02_wctrf_1__20__02__00__00_features_HIDDEN':'0',
                                                    '_1_02_wctrf_1__20__02__00__01_1_INT_K':hp.choice('_1_02_wctrf_1__20__02__00__01_1_INT_K', ['1', ]),
                                                },
                                                {
                                                    '_1_02_wctrf_1__20__02__01__00_features_HIDDEN':'1',
                                                    '_1_02_wctrf_1__20__02__01__01_2_INT_K':hp.qloguniform('_1_02_wctrf_1__20__02__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                                                },
                                            ]),
                                            '_1_02_wctrf_1__20__03_auto_param_302':hp.choice('_1_02_wctrf_1__20__03_auto_param_302', [
                                                {
                                                    '_1_02_wctrf_1__20__03__00__00_depth_HIDDEN':'0',
                                                    '_1_02_wctrf_1__20__03__00__01_1_INT_depth':hp.choice('_1_02_wctrf_1__20__03__00__01_1_INT_depth', ['1', ]),
                                                },
                                                {
                                                    '_1_02_wctrf_1__20__03__01__00_depth_HIDDEN':'1',
                                                    '_1_02_wctrf_1__20__03__01__01_2_INT_depth':hp.quniform('_1_02_wctrf_1__20__03__01__01_2_INT_depth', 2.0, 20.0,1.0),
                                                },
                                            ]),
                                            '_1_02_wctrf_2_QUOTE_END':'REMOVED',
                                        },
                                        {
                                            '_1_02_wctrt_0_QUOTE_START_B':'weka.classifiers.trees.RandomTree',
                                            '_1_02_wctrt_1__21__01_INT_M':hp.qloguniform('_1_02_wctrt_1__21__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                            '_1_02_wctrt_1__21__02_U':hp.choice('_1_02_wctrt_1__21__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                                            '_1_02_wctrt_1__21__03_auto_param_306':hp.choice('_1_02_wctrt_1__21__03_auto_param_306', [
                                                {
                                                    '_1_02_wctrt_1__21__03__00__00_features_HIDDEN':'0',
                                                    '_1_02_wctrt_1__21__03__00__01_1_INT_K':hp.choice('_1_02_wctrt_1__21__03__00__01_1_INT_K', ['0', ]),
                                                },
                                                {
                                                    '_1_02_wctrt_1__21__03__01__00_features_HIDDEN':'1',
                                                    '_1_02_wctrt_1__21__03__01__01_2_INT_K':hp.qloguniform('_1_02_wctrt_1__21__03__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                                                },
                                            ]),
                                            '_1_02_wctrt_1__21__04_auto_param_309':hp.choice('_1_02_wctrt_1__21__04_auto_param_309', [
                                                {
                                                    '_1_02_wctrt_1__21__04__00__00_depth_HIDDEN':'0',
                                                    '_1_02_wctrt_1__21__04__00__01_1_INT_depth':hp.choice('_1_02_wctrt_1__21__04__00__01_1_INT_depth', ['0', ]),
                                                },
                                                {
                                                    '_1_02_wctrt_1__21__04__01__00_depth_HIDDEN':'1',
                                                    '_1_02_wctrt_1__21__04__01__01_2_INT_depth':hp.quniform('_1_02_wctrt_1__21__04__01__01_2_INT_depth', 2.0, 20.0,1.0),
                                                },
                                            ]),
                                            '_1_02_wctrt_1__21__05_auto_param_312':hp.choice('_1_02_wctrt_1__21__05_auto_param_312', [
                                                {
                                                    '_1_02_wctrt_1__21__05__00__00_back_HIDDEN':'0',
                                                    '_1_02_wctrt_1__21__05__00__01_1_INT_N':hp.choice('_1_02_wctrt_1__21__05__00__01_1_INT_N', ['0', ]),
                                                },
                                                {
                                                    '_1_02_wctrt_1__21__05__01__00_back_HIDDEN':'1',
                                                    '_1_02_wctrt_1__21__05__01__01_2_INT_N':hp.quniform('_1_02_wctrt_1__21__05__01__01_2_INT_N', 2.0, 5.0,1.0),
                                                },
                                            ]),
                                            '_1_02_wctrt_2_QUOTE_END':'REMOVED',
                                        },
                                    ]),
                                    hp.choice('auto_param_315', [
                                        (
                                        ),
                                        (
                                            hp.choice('auto_param_318', [
                                                {
                                                    '_1_03_wcbbn_0_QUOTE_START_B':'weka.classifiers.bayes.BayesNet',
                                                    '_1_03_wcbbn_1__00__01_D':hp.choice('_1_03_wcbbn_1__00__01_D', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcbbn_1__00__02_Q':hp.choice('_1_03_wcbbn_1__00__02_Q', ['weka.classifiers.bayes.net.search.local.TAN', 'weka.classifiers.bayes.net.search.local.HillClimber', 'weka.classifiers.bayes.net.search.local.LAGDHillClimber', 'weka.classifiers.bayes.net.search.local.SimulatedAnnealing', 'weka.classifiers.bayes.net.search.local.TabuSearch', 'weka.classifiers.bayes.net.search.local.K2', ]),
                                                    '_1_03_wcbbn_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcbnb_0_QUOTE_START_B':'weka.classifiers.bayes.NaiveBayes',
                                                    '_1_03_wcbnb_1__01__01_K':hp.choice('_1_03_wcbnb_1__01__01_K', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcbnb_1__01__02_D':hp.choice('_1_03_wcbnb_1__01__02_D', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcbnb_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcbnbm_0_QUOTE_START_B':'weka.classifiers.bayes.NaiveBayesMultinomial',
                                                    '_1_03_wcbnbm_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcfl_0_QUOTE_START_B':'weka.classifiers.functions.Logistic',
                                                    '_1_03_wcfl_1__03__01_R':hp.loguniform('_1_03_wcfl_1__03__01_R', math.log(1.0E-12), math.log(10.0)),
                                                    '_1_03_wcfl_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcfmp_0_QUOTE_START_B':'weka.classifiers.functions.MultilayerPerceptron',
                                                    '_1_03_wcfmp_1__04__01_L':hp.uniform('_1_03_wcfmp_1__04__01_L', 0.1, 1.0),
                                                    '_1_03_wcfmp_1__04__02_M':hp.uniform('_1_03_wcfmp_1__04__02_M', 0.1, 1.0),
                                                    '_1_03_wcfmp_1__04__03_B':hp.choice('_1_03_wcfmp_1__04__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcfmp_1__04__04_H':hp.choice('_1_03_wcfmp_1__04__04_H', ['t', 'i', 'o', 'a', ]),
                                                    '_1_03_wcfmp_1__04__05_C':hp.choice('_1_03_wcfmp_1__04__05_C', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcfmp_1__04__06_R':hp.choice('_1_03_wcfmp_1__04__06_R', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcfmp_1__04__07_D':hp.choice('_1_03_wcfmp_1__04__07_D', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcfmp_1__04__08_S':hp.choice('_1_03_wcfmp_1__04__08_S', ['1', ]),
                                                    '_1_03_wcfmp_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcfsgd_0_QUOTE_START_B':'weka.classifiers.functions.SGD',
                                                    '_1_03_wcfsgd_1__05__01_F':hp.choice('_1_03_wcfsgd_1__05__01_F', ['2', '1', '0', ]),
                                                    '_1_03_wcfsgd_1__05__02_L':hp.loguniform('_1_03_wcfsgd_1__05__02_L', math.log(1.0E-5), math.log(0.1)),
                                                    '_1_03_wcfsgd_1__05__03_R':hp.loguniform('_1_03_wcfsgd_1__05__03_R', math.log(1.0E-12), math.log(10.0)),
                                                    '_1_03_wcfsgd_1__05__04_N':hp.choice('_1_03_wcfsgd_1__05__04_N', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcfsgd_1__05__05_M':hp.choice('_1_03_wcfsgd_1__05__05_M', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcfsgd_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcfsmo_0_QUOTE_START_B':'weka.classifiers.functions.SMO',
                                                    '_1_03_wcfsmo_1__06__01_0_C':hp.uniform('_1_03_wcfsmo_1__06__01_0_C', 0.5, 1.5),
                                                    '_1_03_wcfsmo_1__06__02_1_N':hp.choice('_1_03_wcfsmo_1__06__02_1_N', ['2', '1', '0', ]),
                                                    '_1_03_wcfsmo_1__06__03_2_M':hp.choice('_1_03_wcfsmo_1__06__03_2_M', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcfsmo_1__06__04_5_QUOTE_END':hp.choice('_1_03_wcfsmo_1__06__04_5_QUOTE_END', ['REMOVED', ]),
                                                    '_1_03_wcfsmo_1__06__05_auto_param_326':hp.choice('_1_03_wcfsmo_1__06__05_auto_param_326', [
                                                        {
                                                            '_1_03_wcfsmo_1__06__05__00__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.NormalizedPolyKernel',
                                                            '_1_03_wcfsmo_1__06__05__00__01_4_npoly_E':hp.uniform('_1_03_wcfsmo_1__06__05__00__01_4_npoly_E', 0.2, 5.0),
                                                            '_1_03_wcfsmo_1__06__05__00__02_4_npoly_L':hp.choice('_1_03_wcfsmo_1__06__05__00__02_4_npoly_L', ['REMOVE_PREV', 'REMOVED', ]),
                                                        },
                                                        {
                                                            '_1_03_wcfsmo_1__06__05__01__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.PolyKernel',
                                                            '_1_03_wcfsmo_1__06__05__01__01_4_poly_E':hp.uniform('_1_03_wcfsmo_1__06__05__01__01_4_poly_E', 0.2, 5.0),
                                                            '_1_03_wcfsmo_1__06__05__01__02_4_poly_L':hp.choice('_1_03_wcfsmo_1__06__05__01__02_4_poly_L', ['REMOVE_PREV', 'REMOVED', ]),
                                                        },
                                                        {
                                                            '_1_03_wcfsmo_1__06__05__02__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.Puk',
                                                            '_1_03_wcfsmo_1__06__05__02__01_4_puk_S':hp.uniform('_1_03_wcfsmo_1__06__05__02__01_4_puk_S', 0.1, 10.0),
                                                            '_1_03_wcfsmo_1__06__05__02__02_4_puk_O':hp.uniform('_1_03_wcfsmo_1__06__05__02__02_4_puk_O', 0.1, 1.0),
                                                        },
                                                        {
                                                            '_1_03_wcfsmo_1__06__05__03__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.RBFKernel',
                                                            '_1_03_wcfsmo_1__06__05__03__01_4_rbf_G':hp.loguniform('_1_03_wcfsmo_1__06__05__03__01_4_rbf_G', math.log(1.0E-4), math.log(1.0)),
                                                        },
                                                    ]),
                                                    '_1_03_wcfsmo_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcfsl_0_QUOTE_START_B':'weka.classifiers.functions.SimpleLogistic',
                                                    '_1_03_wcfsl_1__07__01_S':hp.choice('_1_03_wcfsl_1__07__01_S', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcfsl_1__07__02_A':hp.choice('_1_03_wcfsl_1__07__02_A', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcfsl_1__07__03_auto_param_332':hp.choice('_1_03_wcfsl_1__07__03_auto_param_332', [
                                                        {
                                                            '_1_03_wcfsl_1__07__03__00__00_W_HIDDEN':'0',
                                                            '_1_03_wcfsl_1__07__03__00__01_1_W':hp.choice('_1_03_wcfsl_1__07__03__00__01_1_W', ['0', ]),
                                                        },
                                                        {
                                                            '_1_03_wcfsl_1__07__03__01__00_W_HIDDEN':'1',
                                                            '_1_03_wcfsl_1__07__03__01__01_2_W':hp.uniform('_1_03_wcfsl_1__07__03__01__01_2_W', 0.0, 1.0),
                                                        },
                                                    ]),
                                                    '_1_03_wcfsl_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcfvp_0_QUOTE_START_B':'weka.classifiers.functions.VotedPerceptron',
                                                    '_1_03_wcfvp_1__08__01_INT_I':hp.quniform('_1_03_wcfvp_1__08__01_INT_I', 1.0, 10.0,1.0),
                                                    '_1_03_wcfvp_1__08__02_INT_M':hp.qloguniform('_1_03_wcfvp_1__08__02_INT_M', math.log(5000.0), math.log(50000.0), 1.0),
                                                    '_1_03_wcfvp_1__08__03_E':hp.uniform('_1_03_wcfvp_1__08__03_E', 0.2, 5.0),
                                                    '_1_03_wcfvp_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wclib_0_QUOTE_START_B':'weka.classifiers.lazy.IBk',
                                                    '_1_03_wclib_1__09__01_E':hp.choice('_1_03_wclib_1__09__01_E', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wclib_1__09__02_INT_K':hp.qloguniform('_1_03_wclib_1__09__02_INT_K', math.log(1.0), math.log(64.0), 1.0),
                                                    '_1_03_wclib_1__09__03_X':hp.choice('_1_03_wclib_1__09__03_X', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wclib_1__09__04_F':hp.choice('_1_03_wclib_1__09__04_F', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wclib_1__09__05_I':hp.choice('_1_03_wclib_1__09__05_I', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wclib_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wclks_0_QUOTE_START_B':'weka.classifiers.lazy.KStar',
                                                    '_1_03_wclks_1__10__01_INT_B':hp.quniform('_1_03_wclks_1__10__01_INT_B', 1.0, 100.0,1.0),
                                                    '_1_03_wclks_1__10__02_E':hp.choice('_1_03_wclks_1__10__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wclks_1__10__03_M':hp.choice('_1_03_wclks_1__10__03_M', ['n', 'd', 'm', 'a', ]),
                                                    '_1_03_wclks_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcrdt_0_QUOTE_START_B':'weka.classifiers.rules.DecisionTable',
                                                    '_1_03_wcrdt_1__11__01_E':hp.choice('_1_03_wcrdt_1__11__01_E', ['auc', 'rmse', 'mae', 'acc', ]),
                                                    '_1_03_wcrdt_1__11__02_I':hp.choice('_1_03_wcrdt_1__11__02_I', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcrdt_1__11__03_S':hp.choice('_1_03_wcrdt_1__11__03_S', ['weka.attributeSelection.Ranker', 'weka.attributeSelection.GreedyStepwise', 'weka.attributeSelection.BestFirst', ]),
                                                    '_1_03_wcrdt_1__11__04_X':hp.choice('_1_03_wcrdt_1__11__04_X', ['4', '2', '3', '1', ]),
                                                    '_1_03_wcrdt_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcrjr_0_QUOTE_START_B':'weka.classifiers.rules.JRip',
                                                    '_1_03_wcrjr_1__12__01_N':hp.uniform('_1_03_wcrjr_1__12__01_N', 1.0, 5.0),
                                                    '_1_03_wcrjr_1__12__02_E':hp.choice('_1_03_wcrjr_1__12__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcrjr_1__12__03_P':hp.choice('_1_03_wcrjr_1__12__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcrjr_1__12__04_INT_O':hp.quniform('_1_03_wcrjr_1__12__04_INT_O', 1.0, 5.0,1.0),
                                                    '_1_03_wcrjr_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcror_0_QUOTE_START_B':'weka.classifiers.rules.OneR',
                                                    '_1_03_wcror_1__13__01_INT_B':hp.qloguniform('_1_03_wcror_1__13__01_INT_B', math.log(1.0), math.log(32.0), 1.0),
                                                    '_1_03_wcror_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcrpart_0_QUOTE_START_B':'weka.classifiers.rules.PART',
                                                    '_1_03_wcrpart_1__14__01_INT_N':hp.quniform('_1_03_wcrpart_1__14__01_INT_N', 2.0, 5.0,1.0),
                                                    '_1_03_wcrpart_1__14__02_INT_M':hp.qloguniform('_1_03_wcrpart_1__14__02_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                                    '_1_03_wcrpart_1__14__03_R':hp.choice('_1_03_wcrpart_1__14__03_R', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcrpart_1__14__04_B':hp.choice('_1_03_wcrpart_1__14__04_B', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wcrpart_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wcrzr_0_QUOTE_START_B':'weka.classifiers.rules.ZeroR',
                                                    '_1_03_wcrzr_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wctds_0_QUOTE_START_B':'weka.classifiers.trees.DecisionStump',
                                                    '_1_03_wctds_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wctj_0_QUOTE_START_B':'weka.classifiers.trees.J48',
                                                    '_1_03_wctj_1__17__01_O':hp.choice('_1_03_wctj_1__17__01_O', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctj_1__17__02_U':hp.choice('_1_03_wctj_1__17__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctj_1__17__03_B':hp.choice('_1_03_wctj_1__17__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctj_1__17__04_J':hp.choice('_1_03_wctj_1__17__04_J', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctj_1__17__05_A':hp.choice('_1_03_wctj_1__17__05_A', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctj_1__17__06_S':hp.choice('_1_03_wctj_1__17__06_S', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctj_1__17__07_INT_M':hp.qloguniform('_1_03_wctj_1__17__07_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                                    '_1_03_wctj_1__17__08_C':hp.uniform('_1_03_wctj_1__17__08_C', 0.0, 1.0),
                                                    '_1_03_wctj_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wctlmt_0_QUOTE_START_B':'weka.classifiers.trees.LMT',
                                                    '_1_03_wctlmt_1__18__01_B':hp.choice('_1_03_wctlmt_1__18__01_B', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctlmt_1__18__02_R':hp.choice('_1_03_wctlmt_1__18__02_R', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctlmt_1__18__03_C':hp.choice('_1_03_wctlmt_1__18__03_C', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctlmt_1__18__04_P':hp.choice('_1_03_wctlmt_1__18__04_P', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctlmt_1__18__05_INT_M':hp.qloguniform('_1_03_wctlmt_1__18__05_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                                    '_1_03_wctlmt_1__18__06_A':hp.choice('_1_03_wctlmt_1__18__06_A', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctlmt_1__18__07_auto_param_346':hp.choice('_1_03_wctlmt_1__18__07_auto_param_346', [
                                                        {
                                                            '_1_03_wctlmt_1__18__07__00__00_W_HIDDEN':'0',
                                                            '_1_03_wctlmt_1__18__07__00__01_1_W':hp.choice('_1_03_wctlmt_1__18__07__00__01_1_W', ['0', ]),
                                                        },
                                                        {
                                                            '_1_03_wctlmt_1__18__07__01__00_W_HIDDEN':'1',
                                                            '_1_03_wctlmt_1__18__07__01__01_2_W':hp.uniform('_1_03_wctlmt_1__18__07__01__01_2_W', 0.0, 1.0),
                                                        },
                                                    ]),
                                                    '_1_03_wctlmt_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wctrept_0_QUOTE_START_B':'weka.classifiers.trees.REPTree',
                                                    '_1_03_wctrept_1__19__01_INT_M':hp.qloguniform('_1_03_wctrept_1__19__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                                    '_1_03_wctrept_1__19__02_V':hp.loguniform('_1_03_wctrept_1__19__02_V', math.log(1.0E-5), math.log(0.1)),
                                                    '_1_03_wctrept_1__19__03_P':hp.choice('_1_03_wctrept_1__19__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctrept_1__19__04_auto_param_350':hp.choice('_1_03_wctrept_1__19__04_auto_param_350', [
                                                        {
                                                            '_1_03_wctrept_1__19__04__00__00_depth_HIDDEN':'0',
                                                            '_1_03_wctrept_1__19__04__00__01_1_INT_L':hp.choice('_1_03_wctrept_1__19__04__00__01_1_INT_L', ['-1', ]),
                                                        },
                                                        {
                                                            '_1_03_wctrept_1__19__04__01__00_depth_HIDDEN':'1',
                                                            '_1_03_wctrept_1__19__04__01__01_2_INT_L':hp.quniform('_1_03_wctrept_1__19__04__01__01_2_INT_L', 2.0, 20.0,1.0),
                                                        },
                                                    ]),
                                                    '_1_03_wctrept_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wctrf_0_QUOTE_START_B':'weka.classifiers.trees.RandomForest',
                                                    '_1_03_wctrf_1__20__01_INT_I':hp.qloguniform('_1_03_wctrf_1__20__01_INT_I', math.log(2.0), math.log(256.0), 1.0),
                                                    '_1_03_wctrf_1__20__02_auto_param_354':hp.choice('_1_03_wctrf_1__20__02_auto_param_354', [
                                                        {
                                                            '_1_03_wctrf_1__20__02__00__00_features_HIDDEN':'0',
                                                            '_1_03_wctrf_1__20__02__00__01_1_INT_K':hp.choice('_1_03_wctrf_1__20__02__00__01_1_INT_K', ['1', ]),
                                                        },
                                                        {
                                                            '_1_03_wctrf_1__20__02__01__00_features_HIDDEN':'1',
                                                            '_1_03_wctrf_1__20__02__01__01_2_INT_K':hp.qloguniform('_1_03_wctrf_1__20__02__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                                                        },
                                                    ]),
                                                    '_1_03_wctrf_1__20__03_auto_param_357':hp.choice('_1_03_wctrf_1__20__03_auto_param_357', [
                                                        {
                                                            '_1_03_wctrf_1__20__03__00__00_depth_HIDDEN':'0',
                                                            '_1_03_wctrf_1__20__03__00__01_1_INT_depth':hp.choice('_1_03_wctrf_1__20__03__00__01_1_INT_depth', ['1', ]),
                                                        },
                                                        {
                                                            '_1_03_wctrf_1__20__03__01__00_depth_HIDDEN':'1',
                                                            '_1_03_wctrf_1__20__03__01__01_2_INT_depth':hp.quniform('_1_03_wctrf_1__20__03__01__01_2_INT_depth', 2.0, 20.0,1.0),
                                                        },
                                                    ]),
                                                    '_1_03_wctrf_2_QUOTE_END':'REMOVED',
                                                },
                                                {
                                                    '_1_03_wctrt_0_QUOTE_START_B':'weka.classifiers.trees.RandomTree',
                                                    '_1_03_wctrt_1__21__01_INT_M':hp.qloguniform('_1_03_wctrt_1__21__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                                    '_1_03_wctrt_1__21__02_U':hp.choice('_1_03_wctrt_1__21__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                                                    '_1_03_wctrt_1__21__03_auto_param_361':hp.choice('_1_03_wctrt_1__21__03_auto_param_361', [
                                                        {
                                                            '_1_03_wctrt_1__21__03__00__00_features_HIDDEN':'0',
                                                            '_1_03_wctrt_1__21__03__00__01_1_INT_K':hp.choice('_1_03_wctrt_1__21__03__00__01_1_INT_K', ['0', ]),
                                                        },
                                                        {
                                                            '_1_03_wctrt_1__21__03__01__00_features_HIDDEN':'1',
                                                            '_1_03_wctrt_1__21__03__01__01_2_INT_K':hp.qloguniform('_1_03_wctrt_1__21__03__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                                                        },
                                                    ]),
                                                    '_1_03_wctrt_1__21__04_auto_param_364':hp.choice('_1_03_wctrt_1__21__04_auto_param_364', [
                                                        {
                                                            '_1_03_wctrt_1__21__04__00__00_depth_HIDDEN':'0',
                                                            '_1_03_wctrt_1__21__04__00__01_1_INT_depth':hp.choice('_1_03_wctrt_1__21__04__00__01_1_INT_depth', ['0', ]),
                                                        },
                                                        {
                                                            '_1_03_wctrt_1__21__04__01__00_depth_HIDDEN':'1',
                                                            '_1_03_wctrt_1__21__04__01__01_2_INT_depth':hp.quniform('_1_03_wctrt_1__21__04__01__01_2_INT_depth', 2.0, 20.0,1.0),
                                                        },
                                                    ]),
                                                    '_1_03_wctrt_1__21__05_auto_param_367':hp.choice('_1_03_wctrt_1__21__05_auto_param_367', [
                                                        {
                                                            '_1_03_wctrt_1__21__05__00__00_back_HIDDEN':'0',
                                                            '_1_03_wctrt_1__21__05__00__01_1_INT_N':hp.choice('_1_03_wctrt_1__21__05__00__01_1_INT_N', ['0', ]),
                                                        },
                                                        {
                                                            '_1_03_wctrt_1__21__05__01__00_back_HIDDEN':'1',
                                                            '_1_03_wctrt_1__21__05__01__01_2_INT_N':hp.quniform('_1_03_wctrt_1__21__05__01__01_2_INT_N', 2.0, 5.0,1.0),
                                                        },
                                                    ]),
                                                    '_1_03_wctrt_2_QUOTE_END':'REMOVED',
                                                },
                                            ]),
                                            hp.choice('auto_param_370', [
                                                (
                                                ),
                                                (
                                                    hp.choice('auto_param_373', [
                                                        {
                                                            '_1_04_wcbbn_0_QUOTE_START_B':'weka.classifiers.bayes.BayesNet',
                                                            '_1_04_wcbbn_1__00__01_D':hp.choice('_1_04_wcbbn_1__00__01_D', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcbbn_1__00__02_Q':hp.choice('_1_04_wcbbn_1__00__02_Q', ['weka.classifiers.bayes.net.search.local.TAN', 'weka.classifiers.bayes.net.search.local.HillClimber', 'weka.classifiers.bayes.net.search.local.LAGDHillClimber', 'weka.classifiers.bayes.net.search.local.SimulatedAnnealing', 'weka.classifiers.bayes.net.search.local.TabuSearch', 'weka.classifiers.bayes.net.search.local.K2', ]),
                                                            '_1_04_wcbbn_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcbnb_0_QUOTE_START_B':'weka.classifiers.bayes.NaiveBayes',
                                                            '_1_04_wcbnb_1__01__01_K':hp.choice('_1_04_wcbnb_1__01__01_K', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcbnb_1__01__02_D':hp.choice('_1_04_wcbnb_1__01__02_D', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcbnb_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcbnbm_0_QUOTE_START_B':'weka.classifiers.bayes.NaiveBayesMultinomial',
                                                            '_1_04_wcbnbm_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcfl_0_QUOTE_START_B':'weka.classifiers.functions.Logistic',
                                                            '_1_04_wcfl_1__03__01_R':hp.loguniform('_1_04_wcfl_1__03__01_R', math.log(1.0E-12), math.log(10.0)),
                                                            '_1_04_wcfl_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcfmp_0_QUOTE_START_B':'weka.classifiers.functions.MultilayerPerceptron',
                                                            '_1_04_wcfmp_1__04__01_L':hp.uniform('_1_04_wcfmp_1__04__01_L', 0.1, 1.0),
                                                            '_1_04_wcfmp_1__04__02_M':hp.uniform('_1_04_wcfmp_1__04__02_M', 0.1, 1.0),
                                                            '_1_04_wcfmp_1__04__03_B':hp.choice('_1_04_wcfmp_1__04__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcfmp_1__04__04_H':hp.choice('_1_04_wcfmp_1__04__04_H', ['t', 'i', 'o', 'a', ]),
                                                            '_1_04_wcfmp_1__04__05_C':hp.choice('_1_04_wcfmp_1__04__05_C', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcfmp_1__04__06_R':hp.choice('_1_04_wcfmp_1__04__06_R', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcfmp_1__04__07_D':hp.choice('_1_04_wcfmp_1__04__07_D', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcfmp_1__04__08_S':hp.choice('_1_04_wcfmp_1__04__08_S', ['1', ]),
                                                            '_1_04_wcfmp_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcfsgd_0_QUOTE_START_B':'weka.classifiers.functions.SGD',
                                                            '_1_04_wcfsgd_1__05__01_F':hp.choice('_1_04_wcfsgd_1__05__01_F', ['2', '1', '0', ]),
                                                            '_1_04_wcfsgd_1__05__02_L':hp.loguniform('_1_04_wcfsgd_1__05__02_L', math.log(1.0E-5), math.log(0.1)),
                                                            '_1_04_wcfsgd_1__05__03_R':hp.loguniform('_1_04_wcfsgd_1__05__03_R', math.log(1.0E-12), math.log(10.0)),
                                                            '_1_04_wcfsgd_1__05__04_N':hp.choice('_1_04_wcfsgd_1__05__04_N', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcfsgd_1__05__05_M':hp.choice('_1_04_wcfsgd_1__05__05_M', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcfsgd_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcfsmo_0_QUOTE_START_B':'weka.classifiers.functions.SMO',
                                                            '_1_04_wcfsmo_1__06__01_0_C':hp.uniform('_1_04_wcfsmo_1__06__01_0_C', 0.5, 1.5),
                                                            '_1_04_wcfsmo_1__06__02_1_N':hp.choice('_1_04_wcfsmo_1__06__02_1_N', ['2', '1', '0', ]),
                                                            '_1_04_wcfsmo_1__06__03_2_M':hp.choice('_1_04_wcfsmo_1__06__03_2_M', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcfsmo_1__06__04_5_QUOTE_END':hp.choice('_1_04_wcfsmo_1__06__04_5_QUOTE_END', ['REMOVED', ]),
                                                            '_1_04_wcfsmo_1__06__05_auto_param_381':hp.choice('_1_04_wcfsmo_1__06__05_auto_param_381', [
                                                                {
                                                                    '_1_04_wcfsmo_1__06__05__00__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.NormalizedPolyKernel',
                                                                    '_1_04_wcfsmo_1__06__05__00__01_4_npoly_E':hp.uniform('_1_04_wcfsmo_1__06__05__00__01_4_npoly_E', 0.2, 5.0),
                                                                    '_1_04_wcfsmo_1__06__05__00__02_4_npoly_L':hp.choice('_1_04_wcfsmo_1__06__05__00__02_4_npoly_L', ['REMOVE_PREV', 'REMOVED', ]),
                                                                },
                                                                {
                                                                    '_1_04_wcfsmo_1__06__05__01__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.PolyKernel',
                                                                    '_1_04_wcfsmo_1__06__05__01__01_4_poly_E':hp.uniform('_1_04_wcfsmo_1__06__05__01__01_4_poly_E', 0.2, 5.0),
                                                                    '_1_04_wcfsmo_1__06__05__01__02_4_poly_L':hp.choice('_1_04_wcfsmo_1__06__05__01__02_4_poly_L', ['REMOVE_PREV', 'REMOVED', ]),
                                                                },
                                                                {
                                                                    '_1_04_wcfsmo_1__06__05__02__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.Puk',
                                                                    '_1_04_wcfsmo_1__06__05__02__01_4_puk_S':hp.uniform('_1_04_wcfsmo_1__06__05__02__01_4_puk_S', 0.1, 10.0),
                                                                    '_1_04_wcfsmo_1__06__05__02__02_4_puk_O':hp.uniform('_1_04_wcfsmo_1__06__05__02__02_4_puk_O', 0.1, 1.0),
                                                                },
                                                                {
                                                                    '_1_04_wcfsmo_1__06__05__03__00_3_REG_IGNORE_QUOTE_START_K':'weka.classifiers.functions.supportVector.RBFKernel',
                                                                    '_1_04_wcfsmo_1__06__05__03__01_4_rbf_G':hp.loguniform('_1_04_wcfsmo_1__06__05__03__01_4_rbf_G', math.log(1.0E-4), math.log(1.0)),
                                                                },
                                                            ]),
                                                            '_1_04_wcfsmo_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcfsl_0_QUOTE_START_B':'weka.classifiers.functions.SimpleLogistic',
                                                            '_1_04_wcfsl_1__07__01_S':hp.choice('_1_04_wcfsl_1__07__01_S', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcfsl_1__07__02_A':hp.choice('_1_04_wcfsl_1__07__02_A', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcfsl_1__07__03_auto_param_387':hp.choice('_1_04_wcfsl_1__07__03_auto_param_387', [
                                                                {
                                                                    '_1_04_wcfsl_1__07__03__00__00_W_HIDDEN':'0',
                                                                    '_1_04_wcfsl_1__07__03__00__01_1_W':hp.choice('_1_04_wcfsl_1__07__03__00__01_1_W', ['0', ]),
                                                                },
                                                                {
                                                                    '_1_04_wcfsl_1__07__03__01__00_W_HIDDEN':'1',
                                                                    '_1_04_wcfsl_1__07__03__01__01_2_W':hp.uniform('_1_04_wcfsl_1__07__03__01__01_2_W', 0.0, 1.0),
                                                                },
                                                            ]),
                                                            '_1_04_wcfsl_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcfvp_0_QUOTE_START_B':'weka.classifiers.functions.VotedPerceptron',
                                                            '_1_04_wcfvp_1__08__01_INT_I':hp.quniform('_1_04_wcfvp_1__08__01_INT_I', 1.0, 10.0,1.0),
                                                            '_1_04_wcfvp_1__08__02_INT_M':hp.qloguniform('_1_04_wcfvp_1__08__02_INT_M', math.log(5000.0), math.log(50000.0), 1.0),
                                                            '_1_04_wcfvp_1__08__03_E':hp.uniform('_1_04_wcfvp_1__08__03_E', 0.2, 5.0),
                                                            '_1_04_wcfvp_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wclib_0_QUOTE_START_B':'weka.classifiers.lazy.IBk',
                                                            '_1_04_wclib_1__09__01_E':hp.choice('_1_04_wclib_1__09__01_E', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wclib_1__09__02_INT_K':hp.qloguniform('_1_04_wclib_1__09__02_INT_K', math.log(1.0), math.log(64.0), 1.0),
                                                            '_1_04_wclib_1__09__03_X':hp.choice('_1_04_wclib_1__09__03_X', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wclib_1__09__04_F':hp.choice('_1_04_wclib_1__09__04_F', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wclib_1__09__05_I':hp.choice('_1_04_wclib_1__09__05_I', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wclib_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wclks_0_QUOTE_START_B':'weka.classifiers.lazy.KStar',
                                                            '_1_04_wclks_1__10__01_INT_B':hp.quniform('_1_04_wclks_1__10__01_INT_B', 1.0, 100.0,1.0),
                                                            '_1_04_wclks_1__10__02_E':hp.choice('_1_04_wclks_1__10__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wclks_1__10__03_M':hp.choice('_1_04_wclks_1__10__03_M', ['n', 'd', 'm', 'a', ]),
                                                            '_1_04_wclks_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcrdt_0_QUOTE_START_B':'weka.classifiers.rules.DecisionTable',
                                                            '_1_04_wcrdt_1__11__01_E':hp.choice('_1_04_wcrdt_1__11__01_E', ['auc', 'rmse', 'mae', 'acc', ]),
                                                            '_1_04_wcrdt_1__11__02_I':hp.choice('_1_04_wcrdt_1__11__02_I', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcrdt_1__11__03_S':hp.choice('_1_04_wcrdt_1__11__03_S', ['weka.attributeSelection.Ranker', 'weka.attributeSelection.GreedyStepwise', 'weka.attributeSelection.BestFirst', ]),
                                                            '_1_04_wcrdt_1__11__04_X':hp.choice('_1_04_wcrdt_1__11__04_X', ['4', '2', '3', '1', ]),
                                                            '_1_04_wcrdt_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcrjr_0_QUOTE_START_B':'weka.classifiers.rules.JRip',
                                                            '_1_04_wcrjr_1__12__01_N':hp.uniform('_1_04_wcrjr_1__12__01_N', 1.0, 5.0),
                                                            '_1_04_wcrjr_1__12__02_E':hp.choice('_1_04_wcrjr_1__12__02_E', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcrjr_1__12__03_P':hp.choice('_1_04_wcrjr_1__12__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcrjr_1__12__04_INT_O':hp.quniform('_1_04_wcrjr_1__12__04_INT_O', 1.0, 5.0,1.0),
                                                            '_1_04_wcrjr_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcror_0_QUOTE_START_B':'weka.classifiers.rules.OneR',
                                                            '_1_04_wcror_1__13__01_INT_B':hp.qloguniform('_1_04_wcror_1__13__01_INT_B', math.log(1.0), math.log(32.0), 1.0),
                                                            '_1_04_wcror_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcrpart_0_QUOTE_START_B':'weka.classifiers.rules.PART',
                                                            '_1_04_wcrpart_1__14__01_INT_N':hp.quniform('_1_04_wcrpart_1__14__01_INT_N', 2.0, 5.0,1.0),
                                                            '_1_04_wcrpart_1__14__02_INT_M':hp.qloguniform('_1_04_wcrpart_1__14__02_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                                            '_1_04_wcrpart_1__14__03_R':hp.choice('_1_04_wcrpart_1__14__03_R', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcrpart_1__14__04_B':hp.choice('_1_04_wcrpart_1__14__04_B', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wcrpart_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wcrzr_0_QUOTE_START_B':'weka.classifiers.rules.ZeroR',
                                                            '_1_04_wcrzr_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wctds_0_QUOTE_START_B':'weka.classifiers.trees.DecisionStump',
                                                            '_1_04_wctds_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wctj_0_QUOTE_START_B':'weka.classifiers.trees.J48',
                                                            '_1_04_wctj_1__17__01_O':hp.choice('_1_04_wctj_1__17__01_O', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctj_1__17__02_U':hp.choice('_1_04_wctj_1__17__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctj_1__17__03_B':hp.choice('_1_04_wctj_1__17__03_B', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctj_1__17__04_J':hp.choice('_1_04_wctj_1__17__04_J', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctj_1__17__05_A':hp.choice('_1_04_wctj_1__17__05_A', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctj_1__17__06_S':hp.choice('_1_04_wctj_1__17__06_S', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctj_1__17__07_INT_M':hp.qloguniform('_1_04_wctj_1__17__07_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                                            '_1_04_wctj_1__17__08_C':hp.uniform('_1_04_wctj_1__17__08_C', 0.0, 1.0),
                                                            '_1_04_wctj_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wctlmt_0_QUOTE_START_B':'weka.classifiers.trees.LMT',
                                                            '_1_04_wctlmt_1__18__01_B':hp.choice('_1_04_wctlmt_1__18__01_B', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctlmt_1__18__02_R':hp.choice('_1_04_wctlmt_1__18__02_R', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctlmt_1__18__03_C':hp.choice('_1_04_wctlmt_1__18__03_C', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctlmt_1__18__04_P':hp.choice('_1_04_wctlmt_1__18__04_P', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctlmt_1__18__05_INT_M':hp.qloguniform('_1_04_wctlmt_1__18__05_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                                            '_1_04_wctlmt_1__18__06_A':hp.choice('_1_04_wctlmt_1__18__06_A', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctlmt_1__18__07_auto_param_401':hp.choice('_1_04_wctlmt_1__18__07_auto_param_401', [
                                                                {
                                                                    '_1_04_wctlmt_1__18__07__00__00_W_HIDDEN':'0',
                                                                    '_1_04_wctlmt_1__18__07__00__01_1_W':hp.choice('_1_04_wctlmt_1__18__07__00__01_1_W', ['0', ]),
                                                                },
                                                                {
                                                                    '_1_04_wctlmt_1__18__07__01__00_W_HIDDEN':'1',
                                                                    '_1_04_wctlmt_1__18__07__01__01_2_W':hp.uniform('_1_04_wctlmt_1__18__07__01__01_2_W', 0.0, 1.0),
                                                                },
                                                            ]),
                                                            '_1_04_wctlmt_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wctrept_0_QUOTE_START_B':'weka.classifiers.trees.REPTree',
                                                            '_1_04_wctrept_1__19__01_INT_M':hp.qloguniform('_1_04_wctrept_1__19__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                                            '_1_04_wctrept_1__19__02_V':hp.loguniform('_1_04_wctrept_1__19__02_V', math.log(1.0E-5), math.log(0.1)),
                                                            '_1_04_wctrept_1__19__03_P':hp.choice('_1_04_wctrept_1__19__03_P', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctrept_1__19__04_auto_param_405':hp.choice('_1_04_wctrept_1__19__04_auto_param_405', [
                                                                {
                                                                    '_1_04_wctrept_1__19__04__00__00_depth_HIDDEN':'0',
                                                                    '_1_04_wctrept_1__19__04__00__01_1_INT_L':hp.choice('_1_04_wctrept_1__19__04__00__01_1_INT_L', ['-1', ]),
                                                                },
                                                                {
                                                                    '_1_04_wctrept_1__19__04__01__00_depth_HIDDEN':'1',
                                                                    '_1_04_wctrept_1__19__04__01__01_2_INT_L':hp.quniform('_1_04_wctrept_1__19__04__01__01_2_INT_L', 2.0, 20.0,1.0),
                                                                },
                                                            ]),
                                                            '_1_04_wctrept_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wctrf_0_QUOTE_START_B':'weka.classifiers.trees.RandomForest',
                                                            '_1_04_wctrf_1__20__01_INT_I':hp.qloguniform('_1_04_wctrf_1__20__01_INT_I', math.log(2.0), math.log(256.0), 1.0),
                                                            '_1_04_wctrf_1__20__02_auto_param_409':hp.choice('_1_04_wctrf_1__20__02_auto_param_409', [
                                                                {
                                                                    '_1_04_wctrf_1__20__02__00__00_features_HIDDEN':'0',
                                                                    '_1_04_wctrf_1__20__02__00__01_1_INT_K':hp.choice('_1_04_wctrf_1__20__02__00__01_1_INT_K', ['1', ]),
                                                                },
                                                                {
                                                                    '_1_04_wctrf_1__20__02__01__00_features_HIDDEN':'1',
                                                                    '_1_04_wctrf_1__20__02__01__01_2_INT_K':hp.qloguniform('_1_04_wctrf_1__20__02__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                                                                },
                                                            ]),
                                                            '_1_04_wctrf_1__20__03_auto_param_412':hp.choice('_1_04_wctrf_1__20__03_auto_param_412', [
                                                                {
                                                                    '_1_04_wctrf_1__20__03__00__00_depth_HIDDEN':'0',
                                                                    '_1_04_wctrf_1__20__03__00__01_1_INT_depth':hp.choice('_1_04_wctrf_1__20__03__00__01_1_INT_depth', ['1', ]),
                                                                },
                                                                {
                                                                    '_1_04_wctrf_1__20__03__01__00_depth_HIDDEN':'1',
                                                                    '_1_04_wctrf_1__20__03__01__01_2_INT_depth':hp.quniform('_1_04_wctrf_1__20__03__01__01_2_INT_depth', 2.0, 20.0,1.0),
                                                                },
                                                            ]),
                                                            '_1_04_wctrf_2_QUOTE_END':'REMOVED',
                                                        },
                                                        {
                                                            '_1_04_wctrt_0_QUOTE_START_B':'weka.classifiers.trees.RandomTree',
                                                            '_1_04_wctrt_1__21__01_INT_M':hp.qloguniform('_1_04_wctrt_1__21__01_INT_M', math.log(1.0), math.log(64.0), 1.0),
                                                            '_1_04_wctrt_1__21__02_U':hp.choice('_1_04_wctrt_1__21__02_U', ['REMOVE_PREV', 'REMOVED', ]),
                                                            '_1_04_wctrt_1__21__03_auto_param_416':hp.choice('_1_04_wctrt_1__21__03_auto_param_416', [
                                                                {
                                                                    '_1_04_wctrt_1__21__03__00__00_features_HIDDEN':'0',
                                                                    '_1_04_wctrt_1__21__03__00__01_1_INT_K':hp.choice('_1_04_wctrt_1__21__03__00__01_1_INT_K', ['0', ]),
                                                                },
                                                                {
                                                                    '_1_04_wctrt_1__21__03__01__00_features_HIDDEN':'1',
                                                                    '_1_04_wctrt_1__21__03__01__01_2_INT_K':hp.qloguniform('_1_04_wctrt_1__21__03__01__01_2_INT_K', math.log(2.0), math.log(32.0), 1.0),
                                                                },
                                                            ]),
                                                            '_1_04_wctrt_1__21__04_auto_param_419':hp.choice('_1_04_wctrt_1__21__04_auto_param_419', [
                                                                {
                                                                    '_1_04_wctrt_1__21__04__00__00_depth_HIDDEN':'0',
                                                                    '_1_04_wctrt_1__21__04__00__01_1_INT_depth':hp.choice('_1_04_wctrt_1__21__04__00__01_1_INT_depth', ['0', ]),
                                                                },
                                                                {
                                                                    '_1_04_wctrt_1__21__04__01__00_depth_HIDDEN':'1',
                                                                    '_1_04_wctrt_1__21__04__01__01_2_INT_depth':hp.quniform('_1_04_wctrt_1__21__04__01__01_2_INT_depth', 2.0, 20.0,1.0),
                                                                },
                                                            ]),
                                                            '_1_04_wctrt_1__21__05_auto_param_422':hp.choice('_1_04_wctrt_1__21__05_auto_param_422', [
                                                                {
                                                                    '_1_04_wctrt_1__21__05__00__00_back_HIDDEN':'0',
                                                                    '_1_04_wctrt_1__21__05__00__01_1_INT_N':hp.choice('_1_04_wctrt_1__21__05__00__01_1_INT_N', ['0', ]),
                                                                },
                                                                {
                                                                    '_1_04_wctrt_1__21__05__01__00_back_HIDDEN':'1',
                                                                    '_1_04_wctrt_1__21__05__01__01_2_INT_N':hp.quniform('_1_04_wctrt_1__21__05__01__01_2_INT_N', 2.0, 5.0,1.0),
                                                                },
                                                            ]),
                                                            '_1_04_wctrt_2_QUOTE_END':'REMOVED',
                                                        },
                                                    ]),
                                                ),
                                            ]),
                                        ),
                                    ]),
                                ),
                            ]),
                        ),
                    ]),
                ),
            ),
        ]),
    ]),
)
