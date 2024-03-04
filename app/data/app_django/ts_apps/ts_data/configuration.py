numericalCriteria = (
    ('LessThan', 'Less Than'),
    ('GreaterThan', 'Greater Than'),
    ('EqualTo', 'Equal To'),
    ('Between', 'Between'),
)

categoricalCriteria = (
    ('Contains', 'Contains'),
)

criteriaTypes = {
    'Numerical': numericalCriteria,
    'Categorical': categoricalCriteria,
}
