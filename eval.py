def eval_4class(pred_list, targ_list):
	assert len(pred_list)==len(targ_list), 'len(pred_list)!=len(targ_list)'

	acc = 0.0
	num_test = len(pred_list)
	wrong = 0
	for idx in range(num_test):
		pred = pred_list[idx]
		targ = targ_list[idx]
		for i in range(4):
			if pred[i] != targ[i]:
				wrong += 1
				break
	acc = 1.0*(num_test-wrong)/num_test
	return {'acc': acc}


# def evaluation_4class(prediction, y):  # 4 dim
# 	TP1, FP1, FN1, TN1 = 0, 0, 0, 0
# 	TP2, FP2, FN2, TN2 = 0, 0, 0, 0
# 	TP3, FP3, FN3, TN3 = 0, 0, 0, 0
# 	TP4, FP4, FN4, TN4 = 0, 0, 0, 0
# 	e, RMSE, RMSE1, RMSE2, RMSE3, RMSE4 = 0.000001, 0.0, 0.0, 0.0, 0.0, 0.0
# 	for i in range(len(y)):
# 		y_i, p_i = list(y[i]), list(prediction[i])
# 		##RMSE
# 		for j in range(len(y_i)):
# 			RMSE += (y_i[j] - p_i[j]) ** 2
# 		RMSE1 += (y_i[0] - p_i[0]) ** 2
# 		RMSE2 += (y_i[1] - p_i[1]) ** 2
# 		RMSE3 += (y_i[2] - p_i[2]) ** 2
# 		RMSE4 += (y_i[3] - p_i[3]) ** 2
# 		## Pre, Recall, F
# 		Act = str(y_i.index(max(y_i)) + 1)
# 		Pre = str(p_i.index(max(p_i)) + 1)

# 		# print y_i, p_i
# 		# print Act, Pre
# 		## for class 1
# 		if Act == '1' and Pre == '1': TP1 += 1
# 		if Act == '1' and Pre != '1': FN1 += 1
# 		if Act != '1' and Pre == '1': FP1 += 1
# 		if Act != '1' and Pre != '1': TN1 += 1
# 		## for class 2
# 		if Act == '2' and Pre == '2': TP2 += 1
# 		if Act == '2' and Pre != '2': FN2 += 1
# 		if Act != '2' and Pre == '2': FP2 += 1
# 		if Act != '2' and Pre != '2': TN2 += 1
# 		## for class 3
# 		if Act == '3' and Pre == '3': TP3 += 1
# 		if Act == '3' and Pre != '3': FN3 += 1
# 		if Act != '3' and Pre == '3': FP3 += 1
# 		if Act != '3' and Pre != '3': TN3 += 1
# 		## for class 4
# 		if Act == '4' and Pre == '4': TP4 += 1
# 		if Act == '4' and Pre != '4': FN4 += 1
# 		if Act != '4' and Pre == '4': FP4 += 1
# 		if Act != '4' and Pre != '4': TN4 += 1
# 	## print result
# 	Acc_all = round(float(TP1 + TP2 + TP3 + TP4) / float(len(y) + e), 4)
# 	Acc1 = round(float(TP1 + TN1) / float(TP1 + TN1 + FN1 + FP1 + e), 4)
# 	Prec1 = round(float(TP1) / float(TP1 + FP1 + e), 4)
# 	Recll1 = round(float(TP1) / float(TP1 + FN1 + e), 4)
# 	F1 = round(2 * Prec1 * Recll1 / (Prec1 + Recll1 + e), 4)

# 	Acc2 = round(float(TP2 + TN2) / float(TP2 + TN2 + FN2 + FP2 + e), 4)
# 	Prec2 = round(float(TP2) / float(TP2 + FP2 + e), 4)
# 	Recll2 = round(float(TP2) / float(TP2 + FN2 + e), 4)
# 	F2 = round(2 * Prec2 * Recll2 / (Prec2 + Recll2 + e), 4)

# 	Acc3 = round(float(TP3 + TN3) / float(TP3 + TN3 + FN3 + FP3 + e), 4)
# 	Prec3 = round(float(TP3) / float(TP3 + FP3 + e), 4)
# 	Recll3 = round(float(TP3) / float(TP3 + FN3 + e), 4)
# 	F3 = round(2 * Prec3 * Recll3 / (Prec3 + Recll3 + e), 4)

# 	Acc4 = round(float(TP4 + TN4) / float(TP4 + TN4 + FN4 + FP4 + e), 4)
# 	Prec4 = round(float(TP4) / float(TP4 + FP4 + e), 4)
# 	Recll4 = round(float(TP4) / float(TP4 + FN4 + e), 4)
# 	F4 = round(2 * Prec4 * Recll4 / (Prec4 + Recll4 + e), 4)

# 	microF = round((F1 + F2 + F3 + F4) / 4, 5)
# 	RMSE_all = round((RMSE / len(y)) ** 0.5, 4)
# 	RMSE_all_1 = round((RMSE1 / len(y)) ** 0.5, 4)
# 	RMSE_all_2 = round((RMSE2 / len(y)) ** 0.5, 4)
# 	RMSE_all_3 = round((RMSE3 / len(y)) ** 0.5, 4)
# 	RMSE_all_4 = round((RMSE4 / len(y)) ** 0.5, 4)
# 	RMSE_all_avg = round((RMSE_all_1 + RMSE_all_2 + RMSE_all_3 + RMSE_all_4) / 4, 4)
# 	return ['acc:', Acc_all, 'Favg:', microF, RMSE_all, RMSE_all_avg,
# 	        'C1:', Acc1, Prec1, Recll1, F1,
# 	        'C2:', Acc2, Prec2, Recll2, F2,
# 	        'C3:', Acc3, Prec3, Recll3, F3,
# 	        'C4:', Acc4, Prec4, Recll4, F4]