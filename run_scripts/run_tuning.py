
"""
Script used to run all bookends
"""
import pathlib    #New path functions in python
import os         #Standard os os type of functions
import subprocess #used for run
import time
import itertools

threads = 16
num_runs = 20;
#mat_loc = "/scratch-local/lane-matrices/norm";
mat_loc = "/scratch-local/lane-matrices/new_norm";
spmv_path = "/home/uahpal001/heterogeneous-spmv";
record_file = "/home/uahpal001/ampere_tuning_csr2.csv";


#spmv_default_omp = ["acc-spmv-csr", "acc-spmv-csrk"];
#spmv_default_omp = ["acc-spmv-csrk"];
spmv_default_omp = ["cuda-spmv-csrk/cuda"];
spmv_default_omp_schedule = ["cuda"];
spmv_default_omp_chunk = [1];
#spmv_default_omp_module = ["module load pgi"];
#spmv_default_omp_unmodule = ["module unload pgi"];
srss = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]
#ssrss = [4, 6, 8, 12, 16, 24, 32, 48]
#srss  = [4, 6, 8, 12, 16, 24, 32, 48]

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
            mat_long = os.path.join(mat_loc,mat);
            
            
            #iterate over all schedule
            for sch in spmv_default_omp_schedule:

                #interate over all chunk sizes
                for chunk in spmv_default_omp_chunk:

                    #iterate over chunk sizes
                    #for ssrs in ssrss:
                    for srs in srss:
                    
                        #iterate over all cores
                        print(i + "_" + mat + "_" + sch + "_" + str(srs)) #+ str(ssrs) + "_" + str(srs));

                        fid_rec = open(record_file, "a+");

                        
                        #setup omp schdule
                        os.environ['OMP_PLACES'] = 'cores';
                        os.environ['OMP_PROC_BIND'] = 'true';
                        os.environ['OMP_NUM_THREADS'] = str(threads)
                        os.environ['OMP_SCHEDULE'] = 'guided'
                        os.environ['GOMP_CPU_AFFINITY'] = '0-15:1'
  
                        io_file = temp_dir_loc + "/" + mat + "_" + sch + "_" + str(chunk) + "_" + str(threads) + ".txt";
                        
                        tic = time.clock()
                        try:
                            trun = subprocess.run(spmv_path + "/" + i + "/spmv" + " " + mat_long + " " + str(num_runs) + " " + str(srs), #str(ssrs) + " " + str(srs),
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
                        fid_rec.write(i + ", " + mat + ", " + sch + ", " + str(chunk) + ", " + str(threads) + ", (" + str(srs) + "), " + min_val + ", " +  max_val + ", " + avg_val + ", \n");
                        
                        #print(i + ", " + mat + ", " + sch + ", " + str(chunk) + ", " + str(threads) + ", " + min_val + ", " +  max_val + ", " + avg_val + ", \n");
                        print(i + ", " + mat + ", " + sch  + ", " + str(chunk) + ", " + str(threads) + ", (" + str(srs) + "), " + min_val + ", " +  max_val + ", " + avg_val + ", \n");

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
    

