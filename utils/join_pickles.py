import pickle

def manually_join_pickle():
    path_name = f"../output_reports/hapt/gem_random_1.0_"
    # full_dict = pickle.load(open(path_name + '4_logs.pkl', 'rb'))['reports']
    # full_dict[6] = pickle.load(open(path_name + '6_logs.pkl', 'rb'))['reports'][6]
    # full_dict[10] = pickle.load(open(path_name + '10_logs.pkl', 'rb'))['reports'][10]
    # full_dict[15] = pickle.load(open(path_name + '15_logs.pkl', 'rb'))['reports'][15]
    #
    # detailed_dict = pickle.load(open(path_name + '8_logs.pkl', 'rb'))['detailed_acc']
    # print(detailed_dict)
    sizes = [2,4,6,8,10,15]
    dict_so_far = {'reports': {size: {} for size in sizes}, 'errors': {size: {} for size in sizes}, 'detailed_acc': {size: {} for size in sizes}}
    for size in sizes:
        cur_name = path_name + str(size) + '_logs.pkl'
        pickle_dict = pickle.load(open(cur_name, 'rb'))
        for key in list(pickle_dict.keys()):
            dict_so_far[key].update({size: pickle_dict[key][size]})
        pickle.dump(dict_so_far, open(cur_name, 'wb'))
manually_join_pickle()