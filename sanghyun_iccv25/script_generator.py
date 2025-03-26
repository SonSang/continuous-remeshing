import os

GPUS = [0, 1, 2]
NUM_GPUS = len(GPUS)
INPUT_PATH = '/home/sson/dmesh2/exp4_result/d3/mvrecon/1_iccv'
OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__)) + '/bash/'
LEARNING_RATE = 0.1
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# gather gt mesh files
input_paths = []
for root, dirs, files in os.walk(INPUT_PATH):
    if 'gt_mesh.obj' in files:
        input_paths.append(root)

mesh_files = [os.path.join(ip, 'gt_mesh.obj') for ip in input_paths]
num_mesh_files = len(mesh_files)

num_process_per_gpu = num_mesh_files // NUM_GPUS

for gi in range(NUM_GPUS):
    output_path = OUTPUT_PATH + 'run_gpu' + str(gi) + '.sh'
    gidx = GPUS[gi]

    with open(output_path, 'w') as f:
        for i in range(num_process_per_gpu):
            mesh_file = mesh_files[gi * num_process_per_gpu + i]
            mesh_path = mesh_file #os.path.join(INPUT_PATH, mesh_file)
            mesh_name = mesh_file.split('/')[-3]
            recon_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon_2", mesh_name)
            f.write(f'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={gidx} python sanghyun_cvpr25/test.py -rm {mesh_path} -o {recon_output_path} -lr {LEARNING_RATE} -domain -1\n')
            
        if gi == NUM_GPUS - 1:
            for i in range(num_process_per_gpu * NUM_GPUS, num_mesh_files):
                mesh_file = mesh_files[i]
                mesh_path = mesh_file #os.path.join(INPUT_PATH, mesh_file)
                mesh_name = mesh_file.split('/')[-3]
                recon_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon_2", mesh_name)
                f.write(f'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={gidx} python sanghyun_cvpr25/test.py -rm {mesh_path} -o {recon_output_path} -lr {LEARNING_RATE} -domain -1\n')