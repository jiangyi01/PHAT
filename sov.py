def sov_calculation(observed, prediction):
    # H:3;array0   C:4;array1    E:5array2
    s_list = [[[], []], [[], []], [[], []]]
    s_list_not = [[[], []], [[], []], [[], []]]
    sov_sum = []
    s_list_not_all_type = []
    for type in range(3):
        max_num = []
        min_num = []
        s = [[], []]
        s1 = []
        s2 = []
        type_num = 3 + type
        for index in range(len(observed)):
            if observed[index] == type_num:
                s[0].append(index)
        for index in range(len(prediction)):
            if prediction[index] == type_num:
                s[1].append(index)

        for value in s[0]:
            if len(s1) != 0 and (s1[-1][1] + 1) == value:
                s1[-1][1] = value
            else:
                s1.append([value, value])

        for value in s[1]:
            if len(s2) != 0 and (s2[-1][1] + 1) == value:
                s2[-1][1] = value
            else:
                s2.append([value, value])

        single_1=[True]*len(s1)
        single_2=[True]*len(s2)

        for index_1 in range(len(s1)):
            # single_1.append(True)
            for index_2 in range(len(s2)):
                if s1[index_1][1] >= s2[index_2][0] and s1[index_1][0] <= s2[index_2][1]:
                    s_list[type][0].append(s1[index_1])
                    s_list[type][1].append(s2[index_2])
                    single_1[index_1]=False
                    single_2[index_2]=False

        for index in range(len(single_1)):
            if single_1[index]:
                s_list_not[type][0].append(s1[index])
        for index in range(len(single_2)):
            if single_2[index]:
                s_list_not[type][1].append(s2[index])

        #
        s_list[type][0] = np.unique(np.array(s_list[type][0]), axis=0)
        s_list[type][1] = np.unique(np.array(s_list[type][1]), axis=0)
        s_list_not[type][0] = np.unique(np.array(s_list_not[type][0]), axis=0)
        s_list_not[type][1] = np.unique(np.array(s_list_not[type][1]), axis=0)

        # print("S(i)s1",s_list[type][0])
        # print("S(i)s2",s_list[type][1])
        # print("S'(i)s1",s_list_not[type][0])
        # print("S(i)s2",s_list_not[type][1])

        si_sum = []
        Ni = []
        for value_1 in s_list[type][0]:
            minov = 0
            maxov = 0
            len_s1 = value_1[1] - value_1[0] + 1
            # print(value_1)
            len_s2 = 0
            data_s1_s2 = 0
            for value_2 in s_list[type][1]:
                len_s2 = value_2[1] - value_2[0] + 1
                if value_1[1] >= value_2[0] and value_1[0] <= value_2[1]:
                    Ni.append(len_s1)
                    # print(Ni,len_s1)
                    sov_part_sort = sorted([value_1[0], value_1[1], value_2[0], value_2[1]])
                    # print(sov_part_sort)
                    minov = sov_part_sort[2] - sov_part_sort[1] + 1
                    min_num.append(minov)
                    maxov = sov_part_sort[3] - sov_part_sort[0] + 1
                    max_num.append(maxov)
                    data_s1_s2 = min(maxov - minov, minov, len_s1 // 2, len_s2 // 2)
                    si_sum.append(((minov + data_s1_s2) / maxov) * len_s1)

        s_list_not_type = []
        for value in s_list_not[type][0]:
            s_list_not_type.append(value[1] - value[0] + 1)
        s_list_not_all_type.append(sum(Ni) + sum(s_list_not_type))

        sov_sum.append(sum(si_sum))
    print("SOV:",100 * sum(sov_sum) / sum(s_list_not_all_type))
    return 100 * sum(sov_sum) / sum(s_list_not_all_type)