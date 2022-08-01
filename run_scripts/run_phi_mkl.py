"""
Script used to run all bookends
"""
import pathlib    #New path functions in python
import os         #Standard os os type of functions
import subprocess #used for run
import time
import sys

threads = [34,68,136,272]
#threads = [68]
num_runs = 20;
mat_loc = sys.argv[1] + "/matrices/rcm";
spmv_path = "/home1/07803/tg871024/gpu-spmv-csr";
record_file = "/home1/07803/tg871024/phi_mkl_spmv.csv";


spmv_default_omp = ["mkl-spmv"];
#spmv_default_omp_schedule = ["guided","dynamic"];
#spmv_default_omp_schedule = ["static"];
#spmv_default_omp_chunk = [1];
#spmv_default_omp_module = ["module load pgi"];
#spmv_default_omp_unmodule = ["module unload pgi"];

def default_run():
    """
    These are the default runs
    """

    fid_rec = open(record_file, "a+");


    #For each kernel
    for ki in range(len(spmv_default_omp)):

        i = spmv_default_omp[ki];

        print("-------" + i + "------------");

        #make correct path exists
        temp_dir_loc = spmv_path + "/runs/" + i ;
        temp_dir = pathlib.Path(temp_dir_loc);
        if(not temp_dir.exists()):
            temp_dir.mkdir(parents=True, exist_ok=True);

        #setup the needed modules
        #zrun = os.system(spmv_default_omp_module[ki]);
        
        #print(os.listdir(mat_loc))
        #iterate over all files
        mat_files = os.listdir(mat_loc);
        
        for mat in mat_files:
                  if mat == "rcmnew":
                    continue
                  mat_long = os.path.join(mat_loc,mat);
                  for thread in threads:
                        print(i + "_" + mat + "_" + str(thread));

                        fid_rec = open(record_file, "a+");

                        
                        #setup omp schdule
                        os.environ['MKL_NUM_THREADS'] = str(thread)
                        #os.environ['OMP_SCHEDULE'] = sch + "," + str(chunk)
                        os.environ['KMP_AFFINITY'] = "balanced"
                        #os.environ['MEMKIND_HBW_NODES'] = '1'
                        #os.environ['GOMP_CPU_AFFINITY'] = '0-271:' + str(int(272/thread))
                        #print('0-271:' + str(int(272/thread)))
                        #os.environ['OMP_PROC_BIND'] = "true"
                        #os.environ['OMP_PLACES'] = "threads"
  
                        io_file = temp_dir_loc + "/" + mat + "_" + str(thread) + ".txt";
                        
                        tic = time.clock()
                        try:
                                trun = subprocess.run("numactl --membind=1 " + spmv_path + "/" + i + "/spmv.exe" + " " + mat_long + " " + str(num_runs),
                                                  stdout=subprocess.PIPE,
                                                  shell=True,
                                                  timeout=600);

                        except subprocess.TimeoutExpired:
                            toc = time.clock()
                            print("Timeout: " , mat_long, " " , threads, "Time: ", toc-tic)
                            continue
                        

                        fid = open(io_file, "w");
                        fid.write(trun.stdout.decode("utf-8"));
                        fid.close();
                        

                        #get the time [min, max, avg]
                        out_utf = trun.stdout.decode("utf-8");
                        
                        #print(out_utf)

                        min_loc = out_utf.find("TimeMin:");
                        min_loc_end = out_utf.find("\n",min_loc);
                        min_val = out_utf[min_loc+8:min_loc_end];
                        #print("min found: ", float(min_val));

                        max_loc = out_utf.find("TimeMax:");
                        max_loc_end = out_utf.find("\n",max_loc);
                        max_val = out_utf[max_loc+8:max_loc_end];
                        #print("max found: ", float(max_val));

                        avg_loc = out_utf.find("TimeAvg:");
                        avg_loc_end = out_utf.find("\n",avg_loc);
                        avg_val = out_utf[avg_loc+8:avg_loc_end];
                        #print("avg found: ", float(avg_val));

                        #write out to the 
                        fid_rec.write(i + ", " + mat + ", " + str(thread) + ", " + min_val + ", " +  max_val + ", " + avg_val + ", \n");
                        
                        #print(i + ", " + mat + ", " + sch + ", " + str(chunk) + ", " + str(thread) + ", " + min_val + ", " +  max_val + ", " + avg_val + ", \n");
                        print(i + ", " + mat  + ", " + str(thread) + ", " + min_val + ", " +  max_val + ", " + avg_val + ", \n");

                        fid_rec.close();

        #zrun = os.system(spmv_default_omp_unmodule[ki]);
    fid_rec.close();


def main():
    """
    
    """
    print("Start with default run");
    default_run();
    print("Done with default run");

main()
    

