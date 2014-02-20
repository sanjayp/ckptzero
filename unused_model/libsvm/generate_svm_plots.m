%% Plots/submission for SVM portion, Question 1.

%% Put your written answers here.
clear all
answers{1} = 'histogram intersection kenel works best. Because the features we used are word frequencies, the idea is to characterize the document(mac or win) with different word frequencies: same/similar word frequencies would produce the same result/prediction. So we need a kernel that best characterize the similarity of the documents based on the word frequencies. The intersection naturally characterizes the overlapping of word frequencies(utilizing the idea of bins) of different documents compared with other kernels.';

save('problem_1_answers.mat', 'answers');

%% Load and process the data.

load ../data/windows_vs_mac.mat;
[X Y] = make_sparse(traindata, vocab);
[Xtest Ytest] = make_sparse(testdata, vocab);

%% Bar Plot - comparing error rates of different kernels

% INSTRUCTIONS: Use the KERNEL_LIBSVM function to evaluate each of the
% kernels you mentioned. Then run the line below to save the results to a
% .mat file.

% make handler function for kernel_libsvm
kernel_linear = @(x,x2) kernel_poly(x, x2, 1);
kernel_quadratic = @(x,x2) kernel_poly(x, x2, 2);
kernel_cubic = @(x,x2) kernel_poly(x, x2, 3);
kernel_gaussian = @(x,x2) kernel_gaussian(x, x2, 20);
kernel_intersect = @(x,x2) kernel_intersection(x, x2);
% kernel_libsvm
results.linear = kernel_libsvm(X, Y, Xtest, Ytest, kernel_linear); % ERROR RATE OF LINEAR KERNEL GOES HERE
results.quadratic = kernel_libsvm(X, Y, Xtest, Ytest, kernel_quadratic);% ERROR RATE OF QUADRATIC KERNEL GOES HERE
results.cubic = kernel_libsvm(X, Y, Xtest, Ytest, kernel_cubic);% ERROR RATE OF CUBIC KERNEL GOES HERE
results.gaussian = kernel_libsvm(X, Y, Xtest, Ytest, kernel_gaussian);% ERROR RATE OF GAUSSIAN (SIGMA=20) GOES HERE
results.intersect = kernel_libsvm(X, Y, Xtest, Ytest, kernel_intersect);% ERROR RATE OF INTERSECTION KERNEL GOES HERE

% Makes a bar chart showing the errors of the different algorithms.
algs = fieldnames(results);
for i = 1:numel(algs)
    y(i) = results.(algs{i});
end
bar(y);
set(gca,'XTickLabel', algs);
xlabel('Kernel');
ylabel('Test Error');
title('Kernel Comparisons');

print -djpeg -r72 plot_1.jpg;
