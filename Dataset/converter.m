ABIDE = load('ABIDE_fc.mat').T;

% Write all columns except FC_norm column
left = ABIDE(:, {'Ins','Id', 'label_id', 'label', 'Age', 'Gender', 'Original_set'});
writetable(left, 'left_table.csv');

% Write matrices into csv files, one csv file for each matrix, filename is its Id
for r = 1:height(ABIDE)
    id = cell2mat(table2cell(ABIDE(r, 'Id')));
    path = strcat('FC_norm/', string(id), '.csv');
    mtx = cell2mat(table2cell(ABIDE(r, 'FC_norm')));
    writematrix(mtx, path)
    disp(path);
end
