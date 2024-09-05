import random
import os
from multiprocessing import Pool

from func_timeout import func_set_timeout

@func_set_timeout(600)
def runcmd1(acmd):
    os.system(acmd)

def runcmd(acmd):
    try:
        runcmd1(acmd)
    except:
        pass


if __name__ == "__main__":
    def select_random_lines(input_file, output_file, num_lines):
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
            selected_lines = random.sample(lines, num_lines)
            
        with open(output_file, 'w') as outfile:
            for line in selected_lines:
                outfile.write(line)

    num_lines = 1500
    os.mkdir(f"params/cgmd")
    select_random_lines("params/pos_data", f"params/cgmd/pos", num_lines)
    select_random_lines("params/neg_data", f"params/cgmd/neg", num_lines)
    pos_data = []
    with open(f"./params/cgmd/pos") as rf:
        lines = rf.readlines()
        for i in range(len(lines)):
            pos_data.append(lines[i].strip())

    neg_data = []
    with open(f"./params/cgmd/neg") as rf:
        lines = rf.readlines()
        for i in range(len(lines)):
            neg_data.append(lines[i].strip())

    floder_list = ["pos_data","neg_data"]
    cmd_list = []

    num = 0
    for floder in floder_list:
        for seq in locals()[floder]:
            with open(f"./cgmd_train/{floder}_{num}_1","w") as wf:
                wf.write(f"{floder}_{num}\n")
                wf.write(f"{floder}_{num}\n")
                wf.write(f"params/amber99sb.prm\n")
                for aacid in seq:
                    wf.write(f"{aacid} -60 -50\n")
                wf.write(f"NH2\n\nN\n\n")
            with open(f"./cgmd_train/{floder}_{num}_2","w") as wf:
                wf.write(f"{floder}_{num}.xyz\nparams/amber99sb.prm\n\n")
            with open(f"./cgmd_train/{floder}_{num}_3","w") as wf:
                wf.write(f"{floder}_{num}.xyz_2\nparams/amber99sb.prm\n\n")
            cmd = f"protein < {floder}_{num}_1 && minimize < {floder}_{num}_2 && xyzpdb < {floder}_{num}_3"
            cmd_list.append(cmd)
            num += 1

    os.chdir(f"./cgmd_train")
    pool = Pool(32)
    pool.map(runcmd,cmd_list)
    pool.close()
    pool.join()

    def read_fasta(fasta_file):
        sequences = {}
        with open(fasta_file, 'r') as file:
            sequence_id = None
            sequence_data = []
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if sequence_id is not None:
                        sequences[sequence_id] = ''.join(sequence_data)
                    sequence_id = line[1:]
                    sequence_data = []
                else:
                    sequence_data.append(line)
            if sequence_id is not None:
                sequences[sequence_id] = ''.join(sequence_data)
        return sequences
    os.chdir(f"../")
    predict_data = []

    fasta_file = 'cluster_res'
    sequences = read_fasta(fasta_file)
    cmd_list = []

    for seq_id, seq_data in sequences.items():
        with open(f"./cgmd_predict/predict_{num}_1","w") as wf:
            wf.write(f"predict_{num}\n")
            wf.write(f"predict_{num}\n")
            wf.write(f"params/amber99sb.prm\n")
            for aacid in seq_data:
                wf.write(f"{aacid} -60 -50\n")
            wf.write(f"NH2\n\nN\n\n")
        with open(f"./cgmd_predict/predict_{num}_2","w") as wf:
            wf.write(f"predict_{num}.xyz\nparams/amber99sb.prm\n\n")
        with open(f"./cgmd_predict/predict_{num}_3","w") as wf:
            wf.write(f"predict_{num}.xyz_2\nparams/amber99sb.prm\n\n")
        cmd = f"protein < predict_{num}_1 && minimize < predict_{num}_2 && xyzpdb < predict_{num}_3"
        cmd_list.append(cmd)
        num += 1

    os.chdir(f"./cgmd_predict")
    pool = Pool(32)
    pool.map(runcmd,cmd_list)
    pool.close()
    pool.join()