from typing import Union, Any, Tuple


def assert_type(assertions: "Union[list,Tuple]"):
    """
    use this function to do type checking on runtime
    - pass an array of assertions
      [(var_1, type_1), (var_2, type_2), ...]
      e.g.: [(arg1, int), (arg2, str), ....]
    - nesting e.g.: list[int] is not possible. Instead do (list_arg[0], int)
    """

    if isinstance(assertions, Tuple):
        assertions = [assertions]

    for i in range(len(assertions)):
        assertion = assertions[i]
        assert isinstance(assertion[0], assertion[1]), (
            "\nWrong type was passed! "
            + str(i + 1)
            + "th assertion fails:\n\t-> Variable  should be of type "
            + str(assertion[1]) + " but is of type " + str(type(assertion[0]))
        )
