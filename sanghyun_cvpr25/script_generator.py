import os

NUM_GPUS = 4
INPUT_PATH = '~/dmesh2/dataset/thingi10k'
OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__)) + '/bash/'
LEARNING_RATE = 0.1
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# open files in config path
INPUT_PATH = os.path.abspath(os.path.expanduser(INPUT_PATH))
files_in_config_path = os.listdir(INPUT_PATH)
mesh_files = [f for f in files_in_config_path if (f.endswith('.obj') or f.endswith('.stl'))]
num_mesh_files = len(mesh_files)

num_process_per_gpu = num_mesh_files // NUM_GPUS

for gi in range(NUM_GPUS):
    output_path = OUTPUT_PATH + 'run_gpu' + str(gi) + '.sh'

    with open(output_path, 'w') as f:
        for i in range(num_process_per_gpu):
            mesh_file = mesh_files[gi * num_process_per_gpu + i]
            mesh_path = os.path.join(INPUT_PATH, mesh_file)
            mesh_name = mesh_file.split('.')[0]
            recon_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon_2", mesh_name)
            f.write(f'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={gi} python sanghyun_cvpr25/test.py -rm {mesh_path} -o {recon_output_path} -lr {LEARNING_RATE}\n')
            
        if gi == NUM_GPUS - 1:
            for i in range(num_process_per_gpu * NUM_GPUS, num_mesh_files):
                mesh_file = mesh_files[i]
                mesh_path = os.path.join(INPUT_PATH, mesh_file)
                mesh_name = mesh_file.split('.')[0]
                recon_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon_2", mesh_name)
                f.write(f'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={gi} python sanghyun_cvpr25/test.py -rm {mesh_path} -o {recon_output_path} -lr {LEARNING_RATE}\n')