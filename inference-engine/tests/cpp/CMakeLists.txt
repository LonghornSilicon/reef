add_executable(test_matrix_cpp test_matrix.cpp)
target_link_libraries(test_matrix_cpp PRIVATE inference_engine)
add_test(NAME matrix_cpp COMMAND test_matrix_cpp)

# TODO: Add focused tests for attention, embeddings, and MLP as implemented.
