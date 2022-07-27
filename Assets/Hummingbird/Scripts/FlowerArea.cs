using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Manages a collection of flower plants and attached flowers
/// </summary>
public class FlowerArea : MonoBehaviour
{
    // The diameter of the area where the agent and flower 
    // can be used for observing relative distances from the agent to the flower
    public const float AreaDiameter = 20f;

    // The list of all flower plants in this area(flower plants have multiple flowers)
    private List<GameObject> flowerPlants;

    // A lookup dictionary for looking up a flower from a nectar collider
    private Dictionary<Collider, Flower> nectarFlowerDictionary;

    /// <summary>
    /// The list of all flowers in the area
    /// </summary>  
    public List<Flower> Flowers
    {
        get;
        private set;
    }

    /// <summary>
    /// The list of all HummingBirds in the area
    /// </summary>  
    public List<HummingBirdAgent> HummingBird
    {
        get;
        private set;
    }

    /// <summary>
    /// The list of all Hunter Agents in the area
    /// </summary>  
    public List<HunterAgent> hunters
    {
        get;
        private set;
    }

    /// <summary>
    /// Reset the flowers and flower plants
    /// </summary> 
    public void ResetFlowers(){
        // Rotate each flower plant around the Y axis and subtly around the X and Z
        foreach (GameObject flowerPlant in flowerPlants)
        {
            float xRotation = UnityEngine.Random.Range(-5f, 5f);
            float yRotation = UnityEngine.Random.Range(-180f, 180f);
            float zRotation = UnityEngine.Random.Range(-5f, 5f);
            flowerPlant.transform.localRotation = Quaternion.Euler(xRotation, yRotation, zRotation);
        }

        // Reset each flower
        foreach(Flower flower in Flowers)
        {
            flower.ResetFlower();
        }
    }

    public void ResetHummingBirds()
    {
        foreach(HummingBirdAgent bird in HummingBird)
        {
            bird.ResetBird();
        }
    }

    public void ResetHunterAgents()
    {
        foreach (HunterAgent hunter in hunters)
        {
            hunter.ResetAgent();
        }
    }

    /// <summary>
    /// Gets the <see cref="Flower"/> that the nectar belongs to
    /// <param name="collider">The Nectar Collider</param>
    /// <return>The matching flower</return>
    /// </summary> 
    public Flower GetFlowerFromNectar(Collider collider)
    {
        return nectarFlowerDictionary[collider];
    }


    /// <summary>
    /// Called when the area wakes up
    /// </summary> 
    private void Awake() 
    {
        // Initialize variables 
        flowerPlants = new List<GameObject>();
        nectarFlowerDictionary = new Dictionary<Collider, Flower>();
        Flowers = new List<Flower>();
        HummingBird = new List<HummingBirdAgent>();
        hunters = new List<HunterAgent>();
    }

    /// <summary>
    /// Called when the game starts
    /// </summary> 
    private void Start() 
    {
        // Finds all flowers that are children of this GameObject/Transform
        FindChildFlowers(transform);
        FindChildBirds(transform);
        //FindChildHunters(transform);
    }


    /// <summary>
    /// Recursively finds all flowers and flower plants that are children of a parent transform
    /// <param name="parent">The parent of the children to check</param>
    /// </summary> 
    private void FindChildFlowers(Transform parent)
    { 
        for(int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.CompareTag("flower_plant"))
            {
                // found a flower plant, add it to the flower plants list
                flowerPlants.Add(child.gameObject);
                
                // look for flowers within the flower plant
                FindChildFlowers(child);
            }
            else
            {
                // Not a flower plant, look for a flower component
                Flower flower = child.GetComponent<Flower>();
                if (flower != null)
                {
                    // found a flower, add it to teh flower's list
                    Flowers.Add(flower);

                    // add nectar colliders to the nectar/flower dictionary
                    nectarFlowerDictionary.Add(flower.nectarCollider, flower);
                    // note, there are no floweres that are children of other flowers
                }
                else
                {
                    // flower component not found, so check children
                    FindChildFlowers(child);
                }
            }
        }
    }


    /// <summary>
    /// Recursively finds all humming birds that are children of a parent transform
    /// <param name="parent">The parent of the children to check</param>
    /// </summary>
    private void FindChildBirds(Transform parent)
    {
        for(int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if(child.CompareTag("humming_bird"))
            {
                HummingBird.Add(child.GetComponent<HummingBirdAgent>());
            }
        }
    }


    /// <summary>
    /// Recursively finds all hunter agents that are children of a parent transform
    /// <param name="parent">The parent of the children to check</param>
    /// </summary>
    private void FindChildHunters(Transform parent)
    {
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.CompareTag("hunter_agent"))
            {
                hunters.Add(child.GetComponent<HunterAgent>());
            }
        }
    }

}
