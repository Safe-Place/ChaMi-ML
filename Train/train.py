# Import library required
import datetime, model, time
import tensorflow as tf

# Train step function with tf function to ran TensorFlow eagerly
@tf.function
def train_step(batch):
  with tf.GradientTape() as tape:
    # Anchor and pos/neg images
    X = batch[:-1]
    # Label
    y = batch[-1]

    # Forward prop
    ypred = my_model(X, training=True)
    # Calculate the loss
    loss = binary_loss(y, ypred)
  
  # Calculate the gradients
  grad = tape.gradient(loss, my_model.trainable_variables)
  # Calculate updated weights and apply to model
  optimizer.apply_gradients(zip(grad, my_model.trainable_variables))
  train_loss_metric(loss)
  train_acc_metric.update_state(y, ypred)

  return loss

# do the same w the evaluation step
@tf.function
def test_step(batch):
  # Anchor and pos/neg images
  X = batch[:-1]
  # Label
  y = batch[-1]
  
  # Make prediction
  ypred = my_model(X, training=False)
  loss = binary_loss(y, ypred)
  val_loss_metric(loss)
  val_acc_metric.update_state(y, ypred)

# Train loop function
def train_model(train_data, val_data, epochs):
  # instantiate the model
  my_model = model.create_siamese_model()
  
  # set optimizer and loss function
  optimizer = tf.keras.optimizers.Adam(1e-4)
  binary_loss = tf.losses.BinaryCrossentropy()

  # Set metrics
  train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
  val_loss_metric = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
  train_acc_metric = tf.keras.metrics.BinaryAccuracy('train_accuracy')
  val_acc_metric = tf.keras.metrics.BinaryAccuracy('val_accuracy')

  # Set up summary writers to write the summaries to disk in a different logs directory
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
  test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  test_summary_writer = tf.summary.create_file_writer(test_log_dir)
  
  # Loop epoch
  for epoch in range(1, epochs+1):
    print(f'\nEpoch {epoch}/{epochs}')

    start_time = time.time()

    # Loop batch
    for step, train_batch in enumerate(train_data):
      # Run train step
      loss_train = train_step(train_batch)
      # Log every 7 batches.
      if step % 7 == 0:
        print(
            "Training loss (for one batch) at step %d: %.4f"
            % (step+7, float(loss_train))
        )
        print("Seen so far: %d samples" % ((step+7) * 16))
    
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    with train_summary_writer.as_default():
      tf.summary.scalar('loss', train_loss_metric.result(), step=epoch)
      tf.summary.scalar('accuracy', train_acc, step=epoch)

    # Run a validation loop at the end of each epoch.
    for val_batch in val_data:
      test_step(val_batch)
    
    val_acc = val_acc_metric.result()

    with test_summary_writer.as_default():
      tf.summary.scalar('loss', val_loss_metric.result(), step=epoch)
      tf.summary.scalar('accuracy', val_acc, step=epoch)

    print("Validation acc: %.4f" % (float(val_acc),))

    print("Time taken: %.2fs" % (time.time() - start_time))
    
    # Reset metrics every epoch
    train_loss_metric.reset_states()
    train_acc_metric.reset_states()
    val_loss_metric.reset_states()
    val_acc_metric.reset_states()
    
    return my_model
