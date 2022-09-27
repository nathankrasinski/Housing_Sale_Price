FLAGS <- flags(
  flag_numeric("nodes", 128),
  flag_numeric("batch_size", 100),
  flag_string("activation1", "softmax"),
  flag_string("activation2", "sigmoid"),
  flag_numeric("learning_rate", 0.01),
  flag_numeric("epochs", 30),
  flag_numeric("dropout1", .5),
  flag_numeric("dropout2", .5)
)

model =keras_model_sequential()
model %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation1, input_shape = ncol(train.onehot)) %>%
  layer_dropout(FLAGS$dropout1) %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation2) %>%
  layer_dropout(FLAGS$dropout2) %>%
  layer_dense(units = 4, activation = "softmax")

model %>% compile(
  optimizer = optimizer_adam(learning_rate=FLAGS$learning_rate),
  loss = 'categorical_crossentropy',
  metrics = 'accuracy')

model %>% fit(
  as.matrix(train.onehot), train.label, epochs = FLAGS$epochs, 
  batch_size = FLAGS$batch_size, validation_data=list(as.matrix(val.onehot), val.label))