import os
from datetime import datetime
import argparse
import uuid

# ==== SWEEP PARAMS ====
SEEDS = [1, 100, 1000]
ENVS = ['cheetah_run', 'walker_walk', 'quadruped_walk', 'cup_catch', 'humanoid_stand']
DEMOS = [10] # TODO dhruv: do 1, 5 later

parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=120)
parser.add_argument('--base_save_dir', default=f'{os.path.abspath(os.path.join(os.getcwd()))}')
parser.add_argument('--output-dirname', default='multijob_iq_output')
parser.add_argument('--error-dirname', default='multijob_iq_error')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--uuid-in-exp-name', action='store_true')
args = parser.parse_args()

# ===== CREATE JOBS ====

jobs = []
params = [(seed, env_name, num_demos) \
    for seed in SEEDS
    for env_name in ENVS
    for num_demos in DEMOS]

for param in params:
    # (seed, task, alg, ref_prop, kl, reward) = param
    # (seed, task, alg, reward, clip) = param
    seed, env_name, num_demos = param
    if args.uuid_in_exp_name:
        id = uuid.uuid4()
    else:
        id = 'NONE'
    name = f'iq_learn_{env_name}_{seed}_demos_{num_demos}_id_{id}'
    
    loss = 'value' if env_name in ['cheetah_run', 'cup_catch'] else 'v0'
    init_temp = 1 if env_name == 'humanoid_stand' else 1e-2

    cmd = 'python train_iq.py '
    cmd += f'env={env_name} seed={seed} agent=sac expert.demos={num_demos} method.loss={loss} method.regularize=True '
    cmd += f'agent.actor_lr=3e-05 agent.init_temp={init_temp} '

    jobs.append((cmd, name, param))

# this you can also hardcode
output_dir = os.path.join(args.base_save_dir, args.output_dirname)
error_dir = os.path.join(args.base_save_dir, args.error_dirname)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)

if not os.path.exists(error_dir):
    os.makedirs(error_dir)
print("Error Directory: %s" % error_dir)

id_name = uuid.uuid4()
now_name = f'{output_dir}/now_{id_name}.txt'
was_name = f'{output_dir}/was_{id_name}.txt'
log_name = f'{output_dir}/log_{id_name}.txt'
err_name = f'{output_dir}/err_{id_name}.txt'
num_commands = 0
jobs = iter(jobs)
done = False
threshold = 999

while not done:
    for (cmd, name, params) in jobs:

        if os.path.exists(now_name):
            file_logic = 'a'  # append if already exists
        else:
            file_logic = 'w'  # make a new file if not
            print(f'creating new file: {now_name}')

        with open(now_name, 'a') as nowfile,\
             open(was_name, 'a') as wasfile,\
             open(log_name, 'a') as output_namefile,\
             open(err_name, 'a') as error_namefile:

            if nowfile.tell() == 0:
                print(f'a new file or the file was empty: {now_name}')

            now = datetime.now()
            datetimestr = now.strftime("%m%d_%H%M:%S.%f")

            num_commands += 1
            nowfile.write(f'{cmd}\n')
            wasfile.write(f'{cmd}\n')

            output_dir = os.path.join(args.base_save_dir, args.output_dirname)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
            error_namefile.write(f'{(os.path.join(error_dir, name))}.error\n')
            if num_commands == threshold:
                break
    
    if num_commands != threshold:
        done = True


    # Make a {name}.slurm file in the {output_dir} which defines this job.
    #slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
    start=1
    slurm_script_path = os.path.join(output_dir, f'iq_{start}_{num_commands}.slurm')
    slurm_command = "sbatch %s" % slurm_script_path

    # Make the .slurm file
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write(f"#SBATCH --array=1-{num_commands}\n")
        slurmfile.write("#SBATCH -o /home/ds844/IQ-Learn-benchmark/slurm/output/iq_%j.out\n")
        slurmfile.write("#SBATCH -e /home/ds844/IQ-Learn-benchmark/slurm/error/iq_%j.err\n")
        # slurmfile.write("#SBATCH --exclude=g2-cpu-01,g2-cpu-02,g2-cpu-03,g2-cpu-04,g2-cpu-05,g2-cpu-06,g2-cpu-07,g2-cpu-08,g2-cpu-09,g2-cpu-10,g2-cpu-11,g2-cpu-25,g2-cpu-26,g2-cpu-27,g2-cpu-28,g2-cpu-29,g2-cpu-30,g2-cpu-97,g2-cpu-98,g2-cpu-99\n")
        slurmfile.write("#SBATCH --requeue\n")
        slurmfile.write("#SBATCH -t %d:00:00\n" % args.nhrs)
        
        # cores requested
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH -n 1\n")
        slurmfile.write("#SBATCH --mem=30G\n")

        # Greene
        # slurmfile.write("#SBATCH --qos gpu48\n")
        slurmfile.write("#SBATCH --partition=default_partition\n")

        slurmfile.write("\n")
        slurmfile.write("source /share/apps/anaconda3/2021.11/etc/profile.d/conda.sh\n")
        slurmfile.write("cd /home/ds844/IQ-Learn-benchmark/iq_learn\n")
        slurmfile.write("conda activate iq-learn\n")
        slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {log_name} | tail -n 1) --error=$(head -n    $SLURM_ARRAY_TASK_ID {err_name} | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {now_name} | tail -n 1)\n" )
        slurmfile.write("\n")

    if not args.dryrun:
        os.system("%s &" % slurm_command)

    num_commands = 0
    id_name = uuid.uuid4()
    now_name = f'{args.base_save_dir}/output/now_{id_name}.txt'
    was_name = f'{args.base_save_dir}/output/was_{id_name}.txt'
    log_name = f'{args.base_save_dir}/output/log_{id_name}.txt'
    err_name = f'{args.base_save_dir}/output/err_{id_name}.txt'