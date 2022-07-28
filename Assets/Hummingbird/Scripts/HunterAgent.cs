using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class HunterAgent : Agent
{
    // ATTRIBUTES ==============================================================================

    [Tooltip("Force to apply when moving")]
    public float moveForce = 2f;

    [Tooltip("Speed to pitch up or down")]
    public float pitchSpeed = 100f;

    [Tooltip("Speed to rotate around the up axis")]
    public float yawSpeed = 100f;

    [Tooltip("The agent's Camera")]
    public Camera agentCamera;

    [Tooltip("Whether this is training mode or Gameplay mode")]
    public bool trainingMode;

    // The rigid body of the agent
    private Rigidbody rigidBody;

    // The flower area that the agent is in 
    private FlowerArea flowerArea;

    // The nearest hummingbird to the hunter at the moment
    private HummingBirdAgent nearestHummingBirdAgent;

    // Whether the agent is frozen(intentionally not fying)
    // observing the environment, but not taking any actions
    private bool frozen = false;

    // Allows for smoother pitch changes
    private float smoothPitchChange = 0f;

    // Allows for smoother yaw changes
    private float smoothYawChange = 0f;

    // Maximum angle that the bird can pitch up or down
    private const float maxPitchAngle = 80f;

    // The next humming bird that the agent will fly to
    private HummingBirdAgent nextHummingBird;

    public int eliminateCount = 0;

    // The current distance to the nearest bird
    private float  distanceToBird;

    /// INITIALIZE ===================================================================================================================================================

    /// <summary>
    /// Initialize the agent
    /// </summary> 
    public override void Initialize()
    {
        rigidBody = GetComponent<Rigidbody>();
        flowerArea = GetComponentInParent<FlowerArea>();

        // If not training mode, no max step, play forever
        if (!trainingMode) MaxStep = 0;
    }

    /// <summary>
    /// Reset the agent when an episode begins
    /// </summary> 
    public override void OnEpisodeBegin()
    {
        // Zero out velocities so that movement stops before a new episode begins
        rigidBody.velocity = Vector3.zero;
        rigidBody.angularVelocity = Vector3.zero;

        // Reestablish the eliminated count for this bird and collision table reference
        eliminateCount = 0;

        // Move the agent to a new Position
        MoveToSafeRandomPosition(true);
    }

    /// OBSERVATIONS ===================================================================================================================================================

    /// <summary>
    /// Collect vector observations from the environment
    /// <param name="sensor">The vector sensor</param>
    /// </summary> 
    public override void CollectObservations(VectorSensor sensor)
    {
        // If nearest flower is null, observe an empty array and return early
        if (nearestHummingBirdAgent == null)
        {
            sensor.AddObservation(new float[9]);
            return;
        }

        // Observe the agent's local rotation (4 observations)
        sensor.AddObservation(transform.localRotation.normalized); // Quaternion with magitude of 1

        // Get a vector from the beak tip to the nearest humming bird
        Vector3 toNearestHummingBirdAgent = nearestHummingBirdAgent.beakTip.position - rigidBody.position;

        // Observe a normalized vector pointing to the nearest humming bird(3 observations)
        sensor.AddObservation(toNearestHummingBirdAgent.normalized);

        // Observe a dot product that indicates whether the beak tip is in front of the flower(1 observation)
        // (+1 means that the beak tip is directly in front of the humming bird, -1 means directly behind)
        sensor.AddObservation(Vector3.Dot(toNearestHummingBirdAgent.normalized, -nearestHummingBirdAgent.beakTip.position)); // beak to flower and down vector into the flower

        // Observe the relative distance from the beak tip to the flower(1 observation)
        sensor.AddObservation(toNearestHummingBirdAgent.magnitude / FlowerArea.AreaDiameter); // ratio to the flower compared to total island diameter

        // 9 total observations
    }
    /// ACTIONS ===================================================================================================================================================

    /// <summary>
    /// Called when an action is received from either the player or the neural network
    ///
    /// vectorAction[i] represents:
    /// Index 0: move vector x (+1 right, -1 left)
    /// Index 1: move vector y (+1 up, -1 down)
    /// Index 2: move vector z (+1 forward, -1 backward)
    /// Index 3: pitch angle (+1 pitch up, -1 pitch down)
    /// Index 4: yaw angle (+1 turn right, -1 turn left)
    /// <paran name="actionBuffers">The actions to take</param>
    /// </summary> 

    public override void OnActionReceived(ActionBuffers vectorAction)
    {
        // Don't take actions if frozen
        if (frozen) return;

        var continuousActions = vectorAction.ContinuousActions;

        // calculate movement vector
        Vector3 move = new(continuousActions[0], continuousActions[1], continuousActions[2]);

        // add force in the direction of the move vector
        rigidBody.AddForce(move * moveForce);

        // Get the current rotation
        Vector3 rotationVector = transform.rotation.eulerAngles;

        // calculate pitch and yaw rotation 
        float pitchChange = continuousActions[3];
        float yawChange = continuousActions[4];

        // calculate mooth rotation changes
        smoothPitchChange = Mathf.MoveTowards(smoothPitchChange, pitchChange, 2f * Time.fixedDeltaTime);
        smoothYawChange = Mathf.MoveTowards(smoothYawChange, yawChange, 2f * Time.fixedDeltaTime);

        // calculate new pitch and yaw based on new smoothed values
        // clamp pitch to avoid flipping upside down
        float pitch = rotationVector.x + smoothPitchChange * Time.fixedDeltaTime * pitchSpeed;
        if (pitch > 180) pitch -= 360f;
        pitch = Mathf.Clamp(pitch, -maxPitchAngle, maxPitchAngle);

        float yaw = rotationVector.y + smoothYawChange * Time.fixedDeltaTime * yawSpeed;

        // apply the new rotation
        transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
    }

    /// <summary>
    /// When behavior type is set to hueristic only on the agent's behavior parameters,
    /// this function will be called. It's return values will be fed into
    /// <see cref="OnActionReceived(float[])">Instead of using the neural network</cref>
    /// <param name="actionsOut">An output action array</param>
    /// </summary>
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;

        // Create placeholders for all movement/turning
        Vector3 forward = Vector3.zero;
        Vector3 left = Vector3.zero;
        Vector3 up = Vector3.zero;
        float pitch = 0f;
        float yaw = 0f;

        // Forward - Backward
        if (Input.GetKey(KeyCode.W)) forward = transform.forward;
        else if (Input.GetKey(KeyCode.S)) forward = -transform.forward;

        // Left - Right
        if (Input.GetKey(KeyCode.A)) left = -transform.right;
        else if (Input.GetKey(KeyCode.D)) left = transform.right;

        // Up - Down
        if (Input.GetKey(KeyCode.E)) up = transform.up;
        else if (Input.GetKey(KeyCode.C)) up = -transform.up;

        // Pitch up - Pitch down
        if (Input.GetKey(KeyCode.UpArrow)) pitch = -0.5f;
        else if (Input.GetKey(KeyCode.DownArrow)) pitch = 0.5f;

        // Turn left - Turn right
        if (Input.GetKey(KeyCode.LeftArrow)) yaw = -0.5f;
        else if (Input.GetKey(KeyCode.RightArrow)) yaw = 0.5f;

        // Combin the movement vectors and normalize
        Vector3 combined = (forward + left + up).normalized;

        continuousActions[0] = combined.x;
        continuousActions[1] = combined.y;
        continuousActions[2] = combined.z;
        continuousActions[3] = pitch;
        continuousActions[4] = yaw;
    }

    /// RIGID BODY ===================================================================================================================================================
  
    /// <summary>
    /// Pevent the agent from moving and taking actions
    /// </summary>
    public void FreezeAgent()
    {
        Debug.Assert(trainingMode == false, "Freeze/Unfreeze not supported in training");
        frozen = true;
        rigidBody.Sleep();
    }

    /// <summary>
    /// Resume Agent movement and actions
    /// </summary>
    public void UnfreezeAgent()
    {
        Debug.Assert(trainingMode == false, "Freeze/Unfreeze not supported in training");
        frozen = false;
        rigidBody.WakeUp();
    }

    /// <summary>
    /// Move the agent into a safe random posotion i.e. does not collide anything 
    /// If also in front of flower, point in front of flower
    /// </summary>
    ///
    private void MoveToSafeRandomPosition(bool inFrontOfHummingBird)
    {
        bool safePositionFound = false;
        int attemptsRemaining = 100; // Prevents infinite loop
        Vector3 potentialPosition = Vector3.zero;
        Quaternion potentialRotation = new();

        // loop until a safe position is found, or we run out of attempts
        while (!safePositionFound && attemptsRemaining > 0)
        {
            attemptsRemaining--;

            // Pick a random flower
            HummingBirdAgent randomBird = flowerArea.HummingBirds[Random.Range(0, flowerArea.HummingBirds.Count)];

            // Position 10-20 cm in from of the flower
            potentialPosition = randomBird.transform.position + randomBird.beakTip.forward * -0.5f;


            //// Pick a random direction rotated around the y axis
            //Quaternion direction = Quaternion.Euler(0f, UnityEngine.Random.Range(-180f, 180f), 0);

            // Choose and set random starting pitch and yaw
            float pitch = UnityEngine.Random.Range(-60f, 60f);
            float yaw = UnityEngine.Random.Range(-180f, 180f);

            potentialRotation = Quaternion.Euler(pitch, yaw, 0f);


            // Check to see if the agent will collide with anything
            Collider[] colliders = Physics.OverlapSphere(potentialPosition, 0.1f);

            // Sage position has been found is no colliders have overlapped
            safePositionFound = colliders.Length == 0;
            
        }
        Debug.Assert(safePositionFound, "Could not find a safe position to spawn");

        transform.SetPositionAndRotation(potentialPosition, potentialRotation);
    }

    /// UPDATE MOTIONS ===========================================================================================================================

    /// <summary>
    /// Called every frame
    /// </summary>
    private void Update()
    {
        if (flowerArea.HummingBirds.Count != 0)
        {
            UpdateNearestHummingBird();
            // Draw a line from the center of the hunter to the nearest humming bird
            Debug.DrawLine(rigidBody.position, nearestHummingBirdAgent.beakTip.position, Color.blue);
        }
    }

    
    /// <summary>
    /// Called every .02 seconds
    /// </summary>
    private void FixedUpdate()
    {
        if (flowerArea.HummingBirds.Count != 0)
        {
            float previousDistance = distanceToBird;
            UpdateNearestHummingBird();
            if (previousDistance > distanceToBird) // if the movements that the bird made make the hunter closer to the bird
            {
                AddReward(0.005f);
            }
            else
            {
                AddReward(-0.005f);
            }
            Debug.DrawLine(rigidBody.position, nearestHummingBirdAgent.beakTip.position, Color.blue);
        }

    }
   

    /// <summary>
    /// Update the nearest Humming bird to the agent
    /// </summary> 
    private void UpdateNearestHummingBird()
    {
        if(nextHummingBird == null )
        {
            nextHummingBird = flowerArea.HummingBirds[UnityEngine.Random.Range(0, flowerArea.HummingBirds.Count)];
        }

        foreach (HummingBirdAgent potentialHummingBird in flowerArea.HummingBirds)
        {
            // TODO
            if (potentialHummingBird.gameObject.activeSelf)
            {
                float distanceToNextHummingBird = Vector3.Distance(nextHummingBird.beakTip.position, transform.position);
                float distanceToPotentialHummingBird = Vector3.Distance(potentialHummingBird.beakTip.position, transform.position);

                // The new distance to the Hummingbird is shorter than the old sitance
                if (distanceToPotentialHummingBird < distanceToNextHummingBird)
                {
                    nextHummingBird = potentialHummingBird;
                }
            }
        }
        
        nearestHummingBirdAgent = nextHummingBird;
        distanceToBird = Vector3.Distance(nearestHummingBirdAgent.beakTip.position, rigidBody.position);
    }

    /// <summary>
    /// Called when the agent's collider enters a trigger collider
    /// </summary>
    /// <param name="other">The trigger collider</param>
    private void OnCollisionEnter(Collision other)
    {
        // Check if agent is colliding with nectar
        if (other.gameObject.CompareTag("humming_bird") && other.gameObject != null)
        {
            // Check to make sure we can aces sinformation correctly
            if (flowerArea.hummingBirdCollision.ContainsKey(other.gameObject))
            {
                // terminates the humming bird if 10 collisions occur
                if ((int)flowerArea.hummingBirdCollision[other.gameObject] == 10)
                {
                    eliminateCount += 1;
                    //flowerArea.HummingBirds.Remove(other.gameObject.GetComponentInParent<HummingBirdAgent>());
                    //Destroy(other.gameObject);

                    other.gameObject.GetComponentInParent<HummingBirdAgent>().gameObject.SetActive(false);
                    Debug.Log(other.gameObject.activeInHierarchy);

                    AddReward(.5f);
                    Debug.Log("Eliminated Bird : " + eliminateCount);

                    nextHummingBird = null;
                    // find a new bird
                    UpdateNearestHummingBird();
                }
                // adds 1 to the collision count
                else
                {
                    flowerArea.hummingBirdCollision[other.gameObject] = (int)(flowerArea.hummingBirdCollision[other.gameObject]) + 1;
                }
            }
            else
            {
                // if the bird is not in the collection, then we start to keep a counter for the object
                flowerArea.hummingBirdCollision.Add(other.gameObject, 1);
            }
            Debug.Log("Collision with bird" + (int)flowerArea.hummingBirdCollision[other.gameObject]);
            AddReward(1f);           
        }

        // Collided with the area boundary, give a negative reward
        if (other.gameObject.CompareTag("boundary"))
        {
            AddReward(-.5f);
        }
    }
}
