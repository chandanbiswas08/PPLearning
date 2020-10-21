#echo "Generating encoded MNIST data"
#python3 mnist-intermediate-layer-op.py

cp data/mnist_slant_frac_vecs.txt data/vecs.txt
cp data/mnist_slant_frac_p.txt data/p.txt
cp data/mnist_slant_frac_s1.txt data/s1.txt
cp data/mnist_slant_frac_s2.txt data/s2.txt
f_name=output/LR_p_gt_pred
python3 Client_primary_task_model.py $f_name
echo "LR: Primary task evaluation (A_P)"
python3 Calculate_accuracy.py $f_name
f_name=output/LR_s1_gt_pred
python3 Adversary_Model.py $f_name S1
echo "LR: Adverserial task S1 evaluation (A_S1)"
python3 Calculate_accuracy.py $f_name
f_name=output/LR_s2_gt_pred
python3 Adversary_Model.py $f_name S2
echo "LR: Adverserial task S2 evaluation (A_S2)"
python3 Calculate_accuracy.py $f_name


gamma1=0.2
gamma2=0.2
cp data/mnist_slant_frac_vecs.txt data/vecs.txt
cp data/mnist_slant_frac_p.txt data/p.txt
cp data/mnist_slant_frac_s1.txt data/s1.txt
cp data/mnist_slant_frac_s2.txt data/s2.txt
echo "gamma1=$gamma1,gamma2=$gamma2"
python3 Multitask_defence.py $gamma1 $gamma2
f_name=output/Multitask_client_Model_p_gt_pred_gamma1_$gamma1
python3 Client_primary_task_model.py $f_name
echo "MT: Primary task evaluation (A_P)"
python3 Calculate_accuracy.py $f_name
f_name=output/Multitask_client_Model_s_gt_pred_gamma1_$gamma1
python3 Adversary_Model.py $f_name S1
echo "MT: Adverserial task S1 evaluation (A_S1)"
python3 Calculate_accuracy.py $f_name
f_name=output/Multitask_client_Model_s1_gt_pred_gamma1_$gamma1
python3 Adversary_Model.py $f_name S2
echo "MT: Adverserial task S2 evaluation (A_S2)"
python3 Calculate_accuracy.py $f_name


mkdir vecfiles
row_len=1
tao=0.4
epochs=300
learning_rate=0.005
vec_file=vecfiles/mnist_single_l2x_tao_$tao"_row_len_"$row_len".vec"
rm -r Weights
cp data/mnist_slant_frac_vecs.txt data/vecs.txt
cp data/mnist_slant_frac_p.txt data/p.txt
cp data/mnist_slant_frac_s1.txt data/s1.txt
cp data/mnist_slant_frac_s2.txt data/s2.txt
python l2x_defence.py train 0.857142857 $tao $vec_file $row_len $epochs $learning_rate
python l2x_defence.py test 0 $tao $vec_file $row_len
echo "l2x_lr row_len=$row_len,tao=$tao"
cp $vec_file data/vecs.txt
f_name=output/single_l2x_p_gt_pred_tao_$tao
python3 Client_primary_task_model.py $f_name
echo "L2X: primary task evaluation (A_P)"
python3 Calculate_accuracy.py $f_name
f_name=output/single_l2x_s1_gt_pred_tao_$tao
python3 Adversary_Model.py $f_name S1
echo "L2X: Adverserial task S1 evaluation (A_S1)"
python3 Calculate_accuracy.py $f_name
f_name=output/single_l2x_s2_gt_pred_tao_$tao
python3 Adversary_Model.py $f_name S2
echo "L2X: Adverserial task S2 evaluation (A_S2)"
python3 Calculate_accuracy.py $f_name

gamma1=0.4
gamma2=0.1
cp $vec_file data/vecs.txt
echo "row_len=$row_len,tao=$tao,gamma1=$gamma1,gamma2=$gamma2"
python3 Multitask_defence.py $gamma1 $gamma2
f_name=output/single_l2x_multitask_p_gt_pred_tao_$tao"_gamma1_"$gamma1"_gamma2_"$gamma2
python3 Client_primary_task_model.py $f_name
echo "L2X-MT: primary task evaluation (A_P)"
python3 Calculate_accuracy.py $f_name
f_name=output/single_l2x_multitask_s1_gt_pred_tao_$tao"_gamma1_"$gamma1"_gamma2_"$gamma2
python3 Adversary_Model.py $f_name S1
echo "L2X-MT: Adverserial task S1 evaluation (A_S1)"
python3 Calculate_accuracy.py $f_name
f_name=output/single_l2x_multitask_s2_gt_pred_tao_$tao"_gamma1_"$gamma1"_gamma2_"$gamma2
python3 Adversary_Model.py $f_name S2
echo "L2X-MT: Adverserial task S2 evaluation (A_S2)"
python3 Calculate_accuracy.py $f_name
