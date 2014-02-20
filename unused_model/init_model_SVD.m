function model = init_model_SVD(vocab)
% Init the SVD adn logistic regression model
% Reference: liblinear package and fsvd function
% 
% Usage:
%
%   model = init_model(vocab)
% 
% Load the SVD matrix and logistic regression model from data folder,  and add some
% necessary dependency(liblinear). 
% The function should take a vocabulary struct as input
%
% EXAMPLES:
%
% >> model = init_model(vocab);
%
% Inint the SVD and logistic regression model

addpath(genpath('./liblinear'));
run ./liblinear/matlab/make;
load ../data/model_lg.mat;
load ../data/model_svd.mat;
% the V matrix
model.V0 = V0;
% pick the most frequent features; excludes those appearing only several
% times
model.high_freq = high_freq;
model.lg = model_lg