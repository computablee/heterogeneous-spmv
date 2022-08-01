matfile = argv(){1};
matrix = fopen(matfile,'r');

m = fscanf(matrix, '%d', 1);
n = fscanf(matrix, '%d', 1);
nnz = fscanf(matrix, '%d', 1);

d = nnz/m;
bx = 8;
by = 12;
vec = false;
veclevel = 4;

ssrs = floor(3.333 + 20 / (d * log(d)) + 0.5);
srs = floor(0.667 * ssrs + 2.667 + 0.5);

if d > 8
    vec = true;
    ssrs = ssrs * 2;
    srs = srs + 2;
end
if d > 16
    veclevel = 8;
    by = 8;
    srs = srs + 1;
end
if d > 32
    veclevel = 16;
    by = 4;
    ssrs = ssrs + 1;
    srs = srs - 1;
end
if d > 64
    veclevel = 32;
    by = 2;
    srs = srs + 1;
end

printf('Optimal calculated super-row size: %d\n', srs);
printf('Optimal calculated super-super-row size: %d\n', ssrs);
printf('Vectorization on innermost loop: ');
if vec
    printf('yes\nOptimal block dimensions: %d x %d x %d\n', veclevel, bx, by);
else
    printf('no\nOptimal block dimensions: %d x %d\n', bx, by);
end
