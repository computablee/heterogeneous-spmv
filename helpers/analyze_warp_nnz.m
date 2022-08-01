matfile = argv(){1};
matrix = fopen(matfile,'r');

ssrs = fscanf(matrix, '%d', 1);
srs = fscanf(matrix, '%d', 1);
m = fscanf(matrix, '%d', 1);
n = fscanf(matrix, '%d', 1);
nnz = fscanf(matrix, '%d', 1);
ssr_ptr = fscanf(matrix, '%d', ssrs);
sr_ptr = fscanf(matrix, '%d', srs);
row_ptr = fscanf(matrix, '%d', m);
col_idx = fscanf(matrix, '%d', nnz);
vals = fscanf(matrix, '%d', nnz);

nnz_per_warp = [];
iters_per_warp = [];
yiters = [];
giters = [];
ctr = 0;
nnz_temp = 0;
iter_temp = 0;

for i = 1:(length(ssr_ptr) - 2)    % gridDim.x
    ostart = ssr_ptr(i + 1);
    oend = ssr_ptr(i + 2);

    giters(length(giters) + 1) = oend - ostart;

    for j = ostart:(oend - 1)      % blockDim.y
        istart = sr_ptr(j + 1);
        iend = sr_ptr(j + 2);

        yiters(length(yiters) + 1) = iend - istart;

        for k = istart:(iend - 1)  % blockDim.x
            rstart = row_ptr(k + 1);
            rend = row_ptr(k + 2);

            nnz_temp = nnz_temp + (rend - rstart);
            iter_temp = iter_temp + 1;
        end

        ctr = ctr + 1;
        if ctr == 4
            nnz_per_warp(length(nnz_per_warp) + 1) = nnz_temp;
            iters_per_warp(length(iters_per_warp) + 1) = iter_temp;
            nnz_temp = 0;
            iter_temp = 0;
            ctr = 0;
        end
    end
end

if ctr > 0
    nnz_per_warp(length(nnz_per_warp) + 1) = nnz_temp;
    iters_per_warp(length(iters_per_warp) + 1) = iter_temp;
end

nnz_avg = mean(nnz_per_warp);
iter_avg = mean(iters_per_warp);
yiter_avg = mean(yiters);
giter_avg = mean(giters);
printf("Average nonzeros per warp is %d\n", nnz_avg);
printf("Average blockDim.x iterations per warp is %d\n", iter_avg);
printf("Average blockDim.y iterations is %d\n", yiter_avg);
printf("Average gridDim.x iterations is %d\n", giter_avg);