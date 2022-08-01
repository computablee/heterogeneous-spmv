function [val, row_ptr, col_ind] = sparse2csr(A)
    m = size(A, 1);
    [col, row, val] = find(A.');
    
    col_ind = col - 1;
    row_ptr = [0; cumsum(histc(row, 1:m))];
end