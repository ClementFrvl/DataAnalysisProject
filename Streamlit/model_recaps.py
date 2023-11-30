models_info = {
    'LinearRegression': {
        'description': 'Utilizes a linear approach to model the relationship between the dependent and independent variables.',
        'key_feature': 'Finds the best-fitting line through the data points.'
    },
    'Ridge': {
        'description': 'A type of linear regression that includes a regularization term to prevent overfitting.',
        'key_feature': 'Uses L2 regularization, adding the squared magnitudes of the coefficients to the loss function.'
    },
    'Lasso': {
        'description': 'Similar to Ridge but uses L1 regularization.',
        'key_feature': 'Encourages sparsity in the coefficient values, effectively selecting a subset of features.'
    },
    'KNeighborsRegressor': {
        'description': 'Non-parametric method that predicts the target variable by averaging the values of its k-nearest neighbors.',
        'key_feature': 'Prediction is based on the majority value among the neighbors.'
    },
    'DecisionTreeRegressor': {
        'description': 'A tree-like model where each node represents a decision based on a feature.',
        'key_feature': 'Splits data recursively until a stopping criterion is met, and predictions are made at the leaves.'
    },
    'RandomForestRegressor': {
        'description': 'Ensemble model that builds multiple decision trees and averages their predictions.',
        'key_feature': 'Reduces overfitting and increases robustness compared to individual trees.'
    },
    'ExtraTreesRegressor': {
        'description': 'Similar to Random Forest but introduces additional randomness in the tree-building process.',
        'key_feature': 'Nodes are split using random thresholds rather than searching for the best possible thresholds.'
    },
    'GradientBoostingRegressor': {
        'description': 'Ensemble method that builds trees sequentially, with each tree correcting the errors of the previous ones.',
        'key_feature': 'Combines weak learners into a strong learner.'
    },
    'SVR': {
        'description': 'Utilizes support vectors to find a hyperplane that best represents the data.',
        'key_feature': 'Seeks to minimize the error while maintaining a margin of tolerance.'
    },
    'LGBMRegressor': {
        'description': 'A gradient boosting framework that uses tree-based learning.',
        'key_feature': 'Optimizes training efficiency by using a histogram-based approach for tree building.'
    },
    'XGBRegressor': {
        'description': 'XGBoost Regressor is an implementation of gradient boosted decision trees designed for speed and performance.',
        'key_feature': 'Optimized for distributed computing and improved regularization to prevent overfitting.'
    }
}
