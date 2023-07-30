"""
I think in case of models we can easily write some general purpose function that will do the job.

But maybe this should be specific to the Stage we are currently at?

1. Turn of weight decay if turned on -> easy to do in optimizer.
2. Don't use fancy optimizers -> stick to Adam (But maybe this depends on stage?)
3. Turn off learning rate decays -> Again, maybe this depends on stage?
4. Set all dropouts to 0
5. Normalization layers probably should stay untouched.
"""


def turn_on_model_regularizations(model, optimizer):
    pass


def turn_off_model_reguralizations(model, optimizer):
    pass
