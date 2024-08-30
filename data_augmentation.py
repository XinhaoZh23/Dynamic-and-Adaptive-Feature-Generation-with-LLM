import numpy as np


def apply_operation(data, op, attr1, attr2=None):
    if attr2 is None:
        new_column_name = f"{attr1}_{op}"
    else:
        new_column_name = f"{attr1}_{op}_{attr2}"

    if op == 'delete':
        if attr1 in data:
            return data.drop(columns=[attr1]), None, new_column_name
        else:
            return data, f"Illegal operation: {attr1} not found", new_column_name

    # Check if the attributes exist in the DataFrame
    if attr1 not in data or (attr2 and attr2 not in data):
        missing_attr = attr1 if attr1 not in data else attr2
        return data, f"Illegal operation: {missing_attr} not found", new_column_name

    # Check for illegal operation if attr2 is None for binary operations
    if op in ['plus', 'subtract', 'multiply', 'divide'] and attr2 is None:
        return data, "Illegal operation: attr2 is required for binary operations", new_column_name

    # Apply binary operations
    if op in ['plus', 'subtract', 'multiply', 'divide'] and attr2:
        if op == 'plus':
            new_column = data[attr1] + data[attr2]
        elif op == 'subtract':
            new_column = data[attr1] - data[attr2]
        elif op == 'multiply':
            new_column = data[attr1] * data[attr2]
        elif op == 'divide':
            new_column = data[attr1] / data[attr2]

        data.insert(len(data.columns) - 1, new_column_name, new_column)

        if data[new_column_name].isna().any():
            data.drop(columns=[new_column_name], inplace=True)
            return data, f"Illegal operation on feature {new_column_name}", new_column_name

        # Check for infinite values
        if data[new_column_name].isin([np.inf, -np.inf]).any():
            # Calculate the number of infinite values
            inf_count = data[new_column_name].isin([np.inf, -np.inf]).sum()

            # If all values are infinite
            if inf_count == len(data[new_column_name]):
                data.drop(columns=[new_column_name], inplace=True)
                return data, f"Illegal operation on feature {new_column_name}", new_column_name
            else:
                # If not all values are infinite, replace them with the mean (excluding infinite values)
                mean_value = data[new_column_name].replace([np.inf, -np.inf], np.nan).mean()
                data[new_column_name].replace([np.inf, -np.inf], mean_value, inplace=True)


        return data, None, new_column_name

    # Apply unary operations
    if op in ['square', 'squareroot', 'cosine', 'sine', 'tangent', 'exp', 'cube', 'log', 'reciprocal', 'sigmoid']:
        if op == 'square':
            new_column = data[attr1] ** 2
        elif op == 'squareroot':
            new_column = np.sqrt(data[attr1])
        elif op == 'cosine':
            new_column = np.cos(data[attr1])
        elif op == 'sine':
            new_column = np.sin(data[attr1])
        elif op == 'tangent':
            new_column = np.tan(data[attr1])
        elif op == 'exp':
            new_column = np.exp(data[attr1])
        elif op == 'cube':
            new_column = data[attr1] ** 3
        elif op == 'log':
            new_column = np.log(data[attr1])
        elif op == 'reciprocal':
            new_column = 1 / data[attr1]
        elif op == 'sigmoid':
            new_column = 1 / (1 + np.exp(-data[attr1]))

        data.insert(len(data.columns) - 1, new_column_name, new_column)

        # Check if there are any NaNs, and return an error if there are
        if data[new_column_name].isna().any():
            data.drop(columns=[new_column_name], inplace=True)
            return data, f"Illegal operation on feature {new_column_name}", new_column_name

        # Check for infinite values
        if data[new_column_name].isin([np.inf, -np.inf]).any():
            # Calculate the number of infinite values
            inf_count = data[new_column_name].isin([np.inf, -np.inf]).sum()

            # If all values are infinite
            if inf_count == len(data[new_column_name]):
                data.drop(columns=[new_column_name], inplace=True)
                return data, f"Illegal operation on feature {new_column_name}", new_column_name
            else:
                # If not all values are infinite, replace them with the mean (excluding infinite values)
                mean_value = data[new_column_name].replace([np.inf, -np.inf], np.nan).mean()
                data[new_column_name].replace([np.inf, -np.inf], mean_value, inplace=True)

        return data, None, new_column_name

    if data[new_column_name].isin([np.inf, -np.inf, np.nan]).any():
        data.drop(columns=[new_column_name], inplace=True)
        return data, f"Illegal operation on feature {new_column_name}", new_column_name

    return data, None, new_column_name

