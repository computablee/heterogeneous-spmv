matfile = argv(){1};
matrix = fopen(matfile, 'r');
type = argv(){2};

if strcmp(type, 'csr3')
  %printf('CSR-3 matrix.\n');
  ssr = fscanf(matrix, '%d', 1);
  sr = fscanf(matrix, '%d', 1);
  m = fscanf(matrix, '%d', 1);
  n = fscanf(matrix, '%d', 1);
  nnz = fscanf(matrix, '%d', 1);

  size = (ssr + sr + m + 2 * nnz) * 4;
elseif strcmp(type, 'csr2')
  %printf('CSR-2 matrix.\n');
  sr = fscanf(matrix, '%d', 1);
  m = fscanf(matrix, '%d', 1);
  n = fscanf(matrix, '%d', 1);
  nnz = fscanf(matrix, '%d', 1);

  size = (sr + m + 2 * nnz) * 4;
elseif strcmp(type, 'csr')
  %printf('CSR matrix.\n');
  m = fscanf(matrix, '%d', 1);
  n = fscanf(matrix, '%d', 1);
  nnz = fscanf(matrix, '%d', 1);

  size = (m + 2 * nnz) * 4;
elseif strcmp(type, 'coo')
  %printf('COO matrix.\n');
  m = fscanf(matrix, '%d', 1);
  n = fscanf(matrix, '%d', 1);
  nnz = fscanf(matrix, '%d', 1);

  size = 3 * nnz * 4;
else
  printf('Unknown matrix type.\n');
end


printf('Matrix consumes %d bytes when loaded into memory.\n', size);