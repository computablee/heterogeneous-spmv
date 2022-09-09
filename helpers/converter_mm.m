myDir = '~/matrices';
myFiles = dir(fullfile(strcat(myDir, '/mm'), '*.mtx'));

for i = 1:length(myFiles)
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(strcat(myDir, '/mm'), baseFileName);
    rcmOutFileName = fullfile(strcat(myDir, '/mm_rcm'), strcat(myFiles(i).name, '.rcm'));
    
    fprintf('Reading matrix %s...', baseFileName)
    [matrix, m, n, numnonzero] = mmread(fullFileName);

    printf('reordering...')    
    tick = tic;
    perm = symrcm(matrix);
    reordered = matrix(perm, perm);
    tock = toc(tick);
    fprintf('reordered in %f...', tock)

    fprintf('writing to file...')
    mmwrite(rcmOutFileName, reordered);

    fprintf('done\n')
end
